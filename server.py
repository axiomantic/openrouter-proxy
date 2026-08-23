#!/usr/bin/env python3
"""
OpenRouter LiteLLM Proxy Service
Provides OpenAI-compatible endpoints with dynamic URL-based configuration
and an interactive web UI for model selection and URL generation.
"""

import os
import sys
import json
import base64
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List, AsyncGenerator
from urllib.parse import parse_qs, unquote

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import litellm
from litellm import acompletion

# Hierarchy of locations for .env:
# 1. Workspace directory (.env)
# 2. ~/.config/openrouter-proxy/.env or ~/.config/openrouter/.env
# 3. ~/.env
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
env_candidates = [
    os.path.join(SCRIPT_DIR, ".env"),
    os.path.expanduser("~/.config/openrouter-proxy/.env"),
    os.path.expanduser("~/.config/openrouter/.env"),
    os.path.expanduser("~/.env"),
]
for env_path in env_candidates:
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("openrouter-proxy")

# Suppress overly verbose litellm logs unless in DEBUG
litellm.suppress_debug_info = True

app = FastAPI(title="OpenRouter LiteLLM Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PORT = int(os.getenv("PORT", "18080"))
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")

# Cache for OpenRouter models list
MODELS_CACHE: Dict[str, Any] = {"timestamp": 0, "data": []}
CACHE_TTL = 300  # 5 minutes


def get_api_key(request: Request) -> str:
    """Extract OpenRouter API key with flexible resolution hierarchy:
    1. Client Authorization Bearer header if it's an OpenRouter key ('sk-or-...')
    2. URL query parameter ('api_key' or 'openrouter_api_key')
    3. Project directory .env / environment variable OPENROUTER_API_KEY
    4. Global ~/.config/openrouter-proxy/.env or ~/.env
    """
    # 1. Check query parameter
    url_key = request.query_params.get("api_key") or request.query_params.get("openrouter_api_key")
    if url_key and url_key.startswith("sk-or-"):
        return url_key

    # 2. Check Auth Header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token.startswith("sk-or-"):
            return token

    # 3. Fallback to server env key
    return os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY)


def decode_config_str(config_str: str) -> Dict[str, Any]:
    """
    Decodes configuration from URL path fragment.
    Supports:
      1. Plain model slug: "anthropic/claude-3.7-sonnet" or "anthropic:claude-3.7-sonnet"
      2. Key-value pairs: "model=openai/gpt-4o,temp=0.7,max_tokens=2048"
      3. Base64-encoded JSON: "eyJtb2RlbCI6ICJnb29nbGUvZ2VtaW5pLTIuNS1wcm8ifQ=="
    """
    if not config_str:
        return {}

    # Try Base64 JSON
    try:
        if config_str.startswith("b64:"):
            raw = base64.urlsafe_b64decode(config_str[4:]).decode("utf-8")
            return json.loads(raw)
        elif len(config_str) > 8 and not any(c in config_str for c in ["/", ":", "=", ","]):
            raw = base64.urlsafe_b64decode(config_str + "==").decode("utf-8")
            return json.loads(raw)
    except Exception:
        pass

    # Try comma-separated key-value pairs (e.g. model=x,temperature=0.7)
    if "=" in config_str:
        result = {}
        for pair in config_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                k = k.strip()
                v = v.strip()
                # Type conversions
                if v.lower() in ("true", "false"):
                    result[k] = v.lower() == "true"
                else:
                    try:
                        if "." in v:
                            result[k] = float(v)
                        else:
                            result[k] = int(v)
                    except ValueError:
                        result[k] = v
        return result

    # Plain model name (colon can replace slash in URL path if needed)
    model = config_str.replace(":", "/")
    return {"model": model}


def extract_url_config(request: Request, path_param: Optional[str] = None) -> Dict[str, Any]:
    """
    Combines path configuration and query parameters into a single dict.
    Query parameters take precedence over path parameters.
    """
    config: Dict[str, Any] = {}

    if path_param:
        config.update(decode_config_str(unquote(path_param)))

    # Parse query parameters
    query_params = request.query_params
    for key, value in query_params.items():
        if value is None or value == "":
            continue
        key_lower = key.lower()

        # Handle numeric/boolean query params
        if key_lower in ("temperature", "temp"):
            try:
                config["temperature"] = float(value)
            except ValueError:
                pass
        elif key_lower in ("top_p", "topp"):
            try:
                config["top_p"] = float(value)
            except ValueError:
                pass
        elif key_lower in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            try:
                config["max_tokens"] = int(value)
            except ValueError:
                pass
        elif key_lower in ("presence_penalty", "frequency_penalty"):
            try:
                config[key_lower] = float(value)
            except ValueError:
                pass
        elif key_lower in ("seed",):
            try:
                config["seed"] = int(value)
            except ValueError:
                pass
        elif key_lower in ("reasoning_effort", "reasoning"):
            config["reasoning_effort"] = value
        elif key_lower in ("system_prompt", "system"):
            config["system_prompt"] = value
        elif key_lower == "model":
            config["model"] = value
        elif key_lower in ("provider_order", "providers"):
            # Comma separated list of providers
            providers = [p.strip() for p in value.split(",") if p.strip()]
            if "extra_body" not in config:
                config["extra_body"] = {}
            if "provider" not in config["extra_body"]:
                config["extra_body"]["provider"] = {}
            config["extra_body"]["provider"]["order"] = providers
        elif key_lower in ("allow_fallbacks", "fallbacks"):
            allow = value.lower() in ("true", "1", "yes")
            if "extra_body" not in config:
                config["extra_body"] = {}
            if "provider" not in config["extra_body"]:
                config["extra_body"]["provider"] = {}
            config["extra_body"]["provider"]["allow_fallbacks"] = allow
        elif key_lower.startswith("extra_"):
            clean_k = key[6:]
            if "extra_body" not in config:
                config["extra_body"] = {}
            config["extra_body"][clean_k] = value
        else:
            config[key] = value

    return config


async def fetch_openrouter_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch and cache available models from OpenRouter."""
    now = time.time()
    if MODELS_CACHE["data"] and (now - MODELS_CACHE["timestamp"] < CACHE_TTL):
        return MODELS_CACHE["data"]

    headers = {}
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{OPENROUTER_BASE_URL}/models", headers=headers)
            if resp.status_code == 200:
                models_data = resp.json().get("data", [])
                MODELS_CACHE["data"] = models_data
                MODELS_CACHE["timestamp"] = now
                logger.info(f"Loaded {len(models_data)} models from OpenRouter")
                return models_data
            else:
                logger.warning(f"Failed to fetch models: HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"Error fetching OpenRouter models: {e}")

    return MODELS_CACHE.get("data", [])


async def handle_proxy_completion(
    request: Request,
    url_config: Dict[str, Any]
) -> Response:
    """Core chat completions handler with streaming and parameter overriding."""
    api_key = get_api_key(request)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "No OpenRouter API key found. Set OPENROUTER_API_KEY in .env or provide Authorization header.",
                    "type": "authentication_error",
                    "code": "missing_api_key"
                }
            }
        )

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Merge configuration: URL config overrides body defaults if specified, or fills in missing body
    # Model resolution priority:
    # 1. URL config 'model'
    # 2. Body 'model'
    # 3. Default fallback
    selected_model = url_config.get("model") or body.get("model") or "anthropic/claude-3.7-sonnet"
    
    # Format model for litellm OpenRouter provider
    if not selected_model.startswith("openrouter/"):
        litellm_model = f"openrouter/{selected_model}"
    else:
        litellm_model = selected_model

    # Extract messages
    messages = body.get("messages", [])
    
    # Prepend or override system prompt if passed in URL config
    if "system_prompt" in url_config and url_config["system_prompt"]:
        sys_prompt = url_config["system_prompt"]
        if messages and messages[0].get("role") == "system":
            # Prepend to existing system prompt or replace
            messages = [{"role": "system", "content": sys_prompt}] + messages[1:]
        else:
            messages = [{"role": "system", "content": sys_prompt}] + messages

    # Build kwargs for litellm
    completion_kwargs: Dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "api_key": api_key,
        "api_base": OPENROUTER_BASE_URL,
    }

    # Handle standard parameters with URL override precedence
    for param in ["temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "seed"]:
        if param in url_config:
            completion_kwargs[param] = url_config[param]
        elif param in body:
            completion_kwargs[param] = body[param]

    # Max completion tokens alternative key
    if "max_completion_tokens" in body and "max_tokens" not in completion_kwargs:
        completion_kwargs["max_tokens"] = body["max_completion_tokens"]

    # Stream flag
    stream = body.get("stream", False)
    if "stream" in url_config:
        stream = bool(url_config["stream"])
    completion_kwargs["stream"] = stream

    # Tools, tool_choice, response_format, stop
    for passthrough_key in ["tools", "tool_choice", "response_format", "stop", "user", "logprobs", "top_logprobs"]:
        if passthrough_key in body:
            completion_kwargs[passthrough_key] = body[passthrough_key]

    # OpenRouter reasoning and extra body configuration
    extra_body = body.get("extra_body", {})
    if "extra_body" in url_config:
        extra_body.update(url_config["extra_body"])

    # Reasoning effort support (e.g. OpenAI o-series / Claude / DeepSeek R1)
    reasoning_effort = url_config.get("reasoning_effort") or body.get("reasoning_effort")
    if reasoning_effort and reasoning_effort != "none":
        if "reasoning" not in extra_body:
            extra_body["reasoning"] = {}
        extra_body["reasoning"]["effort"] = reasoning_effort

    if extra_body:
        completion_kwargs["extra_body"] = extra_body

    logger.info(f"Dispatching completion to {litellm_model} (stream={stream}, temp={completion_kwargs.get('temperature')})")

    # Streaming mode
    if stream:
        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                response = await acompletion(**completion_kwargs)
                async for chunk in response:
                    chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    yield f"data: {json.dumps(chunk_dict)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                err_data = {
                    "error": {
                        "message": str(e),
                        "type": "proxy_error"
                    }
                }
                yield f"data: {json.dumps(err_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    # Non-streaming mode
    try:
        response = await acompletion(**completion_kwargs)
        res_dict = response.model_dump() if hasattr(response, "model_dump") else response
        return JSONResponse(content=res_dict)
    except Exception as e:
        logger.error(f"Completion error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "proxy_error"
                }
            }
        )


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/api/health")
@app.head("/api/health")
async def health_check():
    """Health check and API key readiness check."""
    key = os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    has_key = bool(key and len(key) > 5 and not key.startswith("your_op"))
    masked_key = f"{key[:7]}...{key[-4:]}" if has_key else "Not set / default"
    return {
        "status": "ok",
        "has_api_key": has_key,
        "masked_key": masked_key,
        "port": DEFAULT_PORT,
    }


@app.get("/api/models")
async def get_models(refresh: bool = False):
    """Retrieve available OpenRouter models."""
    if refresh:
        MODELS_CACHE["timestamp"] = 0
    models = await fetch_openrouter_models()
    return {"data": models, "count": len(models)}


@app.get("/v1/models")
@app.get("/p/{config_path:path}/v1/models")
@app.get("/cfg/{b64_config}/v1/models")
async def list_v1_models(request: Request, config_path: Optional[str] = None, b64_config: Optional[str] = None):
    """OpenAI-compatible models list endpoint."""
    models = await fetch_openrouter_models(get_api_key(request))
    openai_models = []
    for m in models:
        openai_models.append({
            "id": m.get("id"),
            "object": "model",
            "created": m.get("created", int(time.time())),
            "owned_by": m.get("id", "").split("/")[0] if "/" in m.get("id", "") else "openrouter",
            "context_length": m.get("context_length", 8192),
            "pricing": m.get("pricing", {}),
            "name": m.get("name", m.get("id")),
            "description": m.get("description", "")
        })
    return {"object": "list", "data": openai_models}


# ------------------------------------------------------------------------------
# Chat completions endpoints supporting query string & path component config
# ------------------------------------------------------------------------------

# 1. Standard OpenAI route: /v1/chat/completions?model=...&temp=...
@app.post("/v1/chat/completions")
async def standard_chat_completions(request: Request):
    url_config = extract_url_config(request)
    return await handle_proxy_completion(request, url_config)


# 2. Path-based route: /p/{config_path}/v1/chat/completions
#    e.g. /p/anthropic:claude-3.7-sonnet/v1/chat/completions
#    e.g. /p/openai:gpt-4o,temp=0.7/v1/chat/completions
@app.post("/p/{config_path:path}/v1/chat/completions")
async def path_chat_completions(request: Request, config_path: str):
    url_config = extract_url_config(request, path_param=config_path)
    return await handle_proxy_completion(request, url_config)


# 3. Base64 configuration route: /cfg/{b64_config}/v1/chat/completions
@app.post("/cfg/{b64_config}/v1/chat/completions")
async def b64_chat_completions(request: Request, b64_config: str):
    url_config = extract_url_config(request, path_param=f"b64:{b64_config}")
    return await handle_proxy_completion(request, url_config)


# 4. Catch-all for clients that append extra subpaths or use /chat/completions directly
@app.post("/chat/completions")
@app.post("/p/{config_path:path}/chat/completions")
async def alternate_chat_completions(request: Request, config_path: Optional[str] = None):
    url_config = extract_url_config(request, path_param=config_path)
    return await handle_proxy_completion(request, url_config)


# ==============================================================================
# Web UI Dashboard
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def web_dashboard(request: Request):
    """Modern single-page configuration generator and live tester."""
    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenRouter LiteLLM Proxy Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 50: '#eef2ff', 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca', 900: '#312e81' },
            dark: { 800: '#1e293b', 850: '#172033', 900: '#0f172a', 950: '#090d16' }
          }
        }
      }
    }
  </script>
  <style>
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
    .badge { @apply inline-flex items-center px-2 py-0.5 rounded text-xs font-medium; }
    .code-block { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  </style>
</head>
<body class="bg-dark-950 text-slate-100 min-h-screen flex flex-col antialiased selection:bg-indigo-500 selection:text-white">

  <!-- Top Navbar -->
  <header class="border-b border-slate-800 bg-dark-900/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/20">
          ⚡
        </div>
        <div>
          <h1 class="font-bold text-lg leading-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
            OpenRouter LiteLLM Proxy
          </h1>
          <p class="text-xs text-slate-400">Dynamic URL Router & Configurator</p>
        </div>
      </div>

      <div class="flex items-center gap-4">
        <div id="status-pill" class="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs text-slate-300">
          <span class="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" id="status-dot"></span>
          <span id="status-text">Checking status...</span>
        </div>
        <button onclick="fetchModels(true)" title="Refresh Models from OpenRouter" class="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
        </button>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">

    <!-- Left Column: Configurator & Model Selector (7 cols) -->
    <div class="lg:col-span-7 flex flex-col gap-6">

      <!-- Model Search & Card Selector -->
      <div class="bg-dark-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-base font-semibold text-white flex items-center gap-2">
            <span>🎯</span> 1. Select Model
          </h2>
          <span id="model-count-badge" class="text-xs text-indigo-400 bg-indigo-950/80 px-2 py-0.5 rounded border border-indigo-800/50">
            Loading models...
          </span>
        </div>

        <!-- Search & Filter Controls -->
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
          <div class="sm:col-span-2 relative">
            <input type="text" id="model-search" placeholder="Search (e.g. claude, gpt-4o, deepseek, llama)..." 
              class="w-full bg-dark-950 border border-slate-700 rounded-xl px-3.5 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500">
            <span class="absolute right-3 top-2.5 text-slate-500 text-xs">⌘K</span>
          </div>
          <div>
            <select id="provider-filter" class="w-full bg-dark-950 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500">
              <option value="all">All Providers</option>
            </select>
          </div>
        </div>

        <!-- Quick Filter Tags -->
        <div class="flex flex-wrap gap-1.5 mb-4 text-xs">
          <button onclick="applyFilterTag('reasoning')" class="filter-tag px-2.5 py-1 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white border border-slate-700">🧠 Reasoning / CoT</button>
          <button onclick="applyFilterTag('vision')" class="filter-tag px-2.5 py-1 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white border border-slate-700">👁️ Vision / Multimodal</button>
          <button onclick="applyFilterTag('free')" class="filter-tag px-2.5 py-1 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white border border-slate-700">🎁 Free / Open</button>
          <button onclick="applyFilterTag('large-context')" class="filter-tag px-2.5 py-1 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white border border-slate-700">📚 100k+ Context</button>
          <button onclick="clearFilterTags()" class="px-2 py-1 rounded-lg text-slate-500 hover:text-slate-300">Reset</button>
        </div>

        <!-- Model Selection List / Carousel Box -->
        <div id="models-list-container" class="max-h-64 overflow-y-auto space-y-2 pr-1 border border-slate-800/80 rounded-xl p-2 bg-dark-950/50">
          <div class="text-center py-8 text-slate-500 text-sm">Fetching model registry from OpenRouter...</div>
        </div>

        <!-- Selected Model Summary Box -->
        <div id="selected-model-card" class="mt-4 p-4 rounded-xl bg-indigo-950/30 border border-indigo-500/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div>
            <div class="flex items-center gap-2">
              <span class="font-bold text-white text-base" id="card-model-name">anthropic/claude-3.7-sonnet</span>
              <span class="px-2 py-0.5 text-xs rounded bg-indigo-600/30 text-indigo-300 font-mono" id="card-context-len">200k ctx</span>
            </div>
            <p class="text-xs text-slate-400 mt-1 line-clamp-1" id="card-model-desc">State-of-the-art hybrid reasoning model</p>
          </div>
          <div class="text-right text-xs text-slate-400">
            <div>Prompt: <span class="text-emerald-400 font-mono" id="card-prompt-price">$3.00/M</span></div>
            <div>Completion: <span class="text-emerald-400 font-mono" id="card-comp-price">$15.00/M</span></div>
          </div>
        </div>
      </div>

      <!-- Parameters & Options Customizer -->
      <div class="bg-dark-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <h2 class="text-base font-semibold text-white mb-4 flex items-center gap-2">
          <span>🎛️</span> 2. Fine-tune Parameters
        </h2>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <!-- Temperature -->
          <div>
            <div class="flex justify-between text-xs mb-1">
              <label class="text-slate-300 font-medium">Temperature</label>
              <span id="temp-val" class="font-mono text-indigo-400 font-bold">0.7</span>
            </div>
            <input type="range" id="param-temp" min="0" max="2" step="0.05" value="0.7" 
              class="w-full accent-indigo-500 bg-slate-800 rounded h-1.5 cursor-pointer">
            <span class="text-[11px] text-slate-500">Lower = focused & deterministic, Higher = creative</span>
          </div>

          <!-- Top P -->
          <div>
            <div class="flex justify-between text-xs mb-1">
              <label class="text-slate-300 font-medium">Top P</label>
              <span id="topp-val" class="font-mono text-indigo-400 font-bold">1.0</span>
            </div>
            <input type="range" id="param-topp" min="0" max="1" step="0.05" value="1.0" 
              class="w-full accent-indigo-500 bg-slate-800 rounded h-1.5 cursor-pointer">
            <span class="text-[11px] text-slate-500">Nucleus sampling threshold</span>
          </div>

          <!-- Max Tokens -->
          <div>
            <div class="flex justify-between text-xs mb-1">
              <label class="text-slate-300 font-medium">Max Output Tokens</label>
              <span id="maxtok-val" class="font-mono text-indigo-400 font-bold">4096</span>
            </div>
            <input type="number" id="param-maxtok" min="1" max="128000" step="256" value="4096" 
              class="w-full bg-dark-950 border border-slate-700 rounded-xl px-3 py-1.5 text-sm text-white focus:outline-none focus:border-indigo-500">
            <span class="text-[11px] text-slate-500">Limit max response length</span>
          </div>

          <!-- Reasoning Effort -->
          <div>
            <div class="flex justify-between text-xs mb-1">
              <label class="text-slate-300 font-medium">Reasoning Effort (CoT)</label>
              <span class="text-[10px] text-violet-400 font-medium">o-series / Claude / R1</span>
            </div>
            <select id="param-reasoning" class="w-full bg-dark-950 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500">
              <option value="none">Disabled / Default</option>
              <option value="low">Low Effort</option>
              <option value="medium">Medium Effort</option>
              <option value="high">High Effort (Deep Reason)</option>
            </select>
          </div>
        </div>

        <!-- Advanced Collapsible Options -->
        <details class="mt-4 pt-4 border-t border-slate-800">
          <summary class="text-xs font-semibold text-slate-400 hover:text-slate-200 cursor-pointer select-none">
            ▸ Advanced: System Prompt Override & OpenRouter Provider Routing
          </summary>
          <div class="mt-3 space-y-3 pt-2">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Custom System Prompt (Prepend/Override)</label>
              <textarea id="param-system-prompt" rows="2" placeholder="e.g. You are an expert software engineer specializing in macOS launchd daemons..." 
                class="w-full bg-dark-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"></textarea>
            </div>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Provider Routing Order (comma-separated)</label>
                <input type="text" id="param-provider-order" placeholder="e.g. Together,Fireworks,DeepInfra" 
                  class="w-full bg-dark-950 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500">
              </div>
              <div class="flex items-center gap-2 pt-4">
                <input type="checkbox" id="param-fallbacks" checked class="rounded bg-dark-950 border-slate-700 text-indigo-600 focus:ring-0">
                <label for="param-fallbacks" class="text-xs text-slate-300">Allow Provider Fallbacks</label>
              </div>
            </div>
          </div>
        </details>
      </div>

    </div>

    <!-- Right Column: Generated Endpoints & Live Tester (5 cols) -->
    <div class="lg:col-span-5 flex flex-col gap-6">

      <!-- Generated URLs Card -->
      <div class="bg-dark-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div class="flex items-center justify-between mb-3">
          <h2 class="text-base font-semibold text-white flex items-center gap-2">
            <span>🔗</span> 3. Generated Proxy URLs
          </h2>
          <span class="text-xs text-emerald-400 bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded">Ready</span>
        </div>
        <p class="text-xs text-slate-400 mb-4">Use these endpoints in Cursor, Continue.dev, Aider, LibreChat, OpenAI SDK, etc.</p>

        <!-- Style 1: Path Based URL (Recommended for most tools) -->
        <div class="mb-4">
          <div class="flex justify-between items-center text-xs mb-1">
            <span class="font-medium text-indigo-300">Path-Based Base URL (Recommended)</span>
            <button onclick="copyToClipboard('path-base-url')" class="text-indigo-400 hover:text-indigo-300 text-xs flex items-center gap-1">
              📋 Copy
            </button>
          </div>
          <div class="p-2.5 rounded-xl bg-dark-950 border border-slate-800 text-xs font-mono text-slate-200 break-all select-all flex items-center justify-between" id="path-base-url">
            http://localhost:18080/p/anthropic:claude-3.7-sonnet/v1
          </div>
          <span class="text-[11px] text-slate-500">Paste as <code class="text-slate-400">OPENAI_BASE_URL</code> or <code class="text-slate-400">baseURL</code></span>
        </div>

        <!-- Style 2: Query String URL -->
        <div class="mb-4">
          <div class="flex justify-between items-center text-xs mb-1">
            <span class="font-medium text-slate-300">Query-String Base URL</span>
            <button onclick="copyToClipboard('query-base-url')" class="text-indigo-400 hover:text-indigo-300 text-xs flex items-center gap-1">
              📋 Copy
            </button>
          </div>
          <div class="p-2.5 rounded-xl bg-dark-950 border border-slate-800 text-xs font-mono text-slate-200 break-all select-all" id="query-base-url">
            http://localhost:18080/v1?model=anthropic/claude-3.7-sonnet&temperature=0.7
          </div>
        </div>

        <!-- Full Completions Endpoint -->
        <div class="mb-4">
          <div class="flex justify-between items-center text-xs mb-1">
            <span class="font-medium text-slate-300">Direct Completions Target</span>
            <button onclick="copyToClipboard('completions-endpoint-url')" class="text-indigo-400 hover:text-indigo-300 text-xs flex items-center gap-1">
              📋 Copy
            </button>
          </div>
          <div class="p-2.5 rounded-xl bg-dark-950 border border-slate-800 text-xs font-mono text-slate-400 break-all select-all" id="completions-endpoint-url">
            http://localhost:18080/p/anthropic:claude-3.7-sonnet/v1/chat/completions
          </div>
        </div>

        <!-- Code Snippet Tabs -->
        <div class="mt-4 border-t border-slate-800 pt-3">
          <div class="flex gap-2 border-b border-slate-800 text-xs pb-1 mb-2">
            <button onclick="setSnippetTab('curl')" id="tab-curl" class="font-semibold text-indigo-400 border-b-2 border-indigo-500 pb-1">cURL</button>
            <button onclick="setSnippetTab('python')" id="tab-python" class="text-slate-400 hover:text-slate-200 pb-1">Python (OpenAI)</button>
            <button onclick="setSnippetTab('js')" id="tab-js" class="text-slate-400 hover:text-slate-200 pb-1">Node.js</button>
            <button onclick="setSnippetTab('aider')" id="tab-aider" class="text-slate-400 hover:text-slate-200 pb-1">Aider / CLI</button>
          </div>

          <div class="relative">
            <pre id="snippet-box" class="p-3 bg-dark-950 border border-slate-800/80 rounded-xl text-[11px] font-mono text-slate-300 overflow-x-auto max-h-48"></pre>
            <button onclick="copySnippet()" class="absolute top-2 right-2 px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[10px]">
              Copy Code
            </button>
          </div>
        </div>
      </div>

      <!-- Live Interactive Prompt Tester -->
      <div class="bg-dark-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex-1 flex flex-col">
        <h2 class="text-base font-semibold text-white mb-2 flex items-center justify-between">
          <span class="flex items-center gap-2"><span>💬</span> 4. Test Proxy Endpoint</span>
          <span id="test-latency" class="text-[11px] font-mono text-slate-400"></span>
        </h2>
        
        <div class="flex gap-2 mb-3">
          <input type="text" id="test-input" placeholder="Say 'Hello!' or test reasoning with 'Write a quick haiku about daemons'..." 
            value="Write a clever haiku about macOS launchd daemons."
            class="flex-1 bg-dark-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500">
          <button id="test-send-btn" onclick="sendTestPrompt()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold shadow-lg shadow-indigo-500/20 transition-all flex items-center gap-1.5">
            <span id="test-btn-icon">🚀</span>
            <span>Send</span>
          </button>
        </div>

        <!-- Response stream container -->
        <div id="test-output" class="flex-1 min-h-[120px] max-h-60 overflow-y-auto p-3 bg-dark-950 border border-slate-800 rounded-xl text-xs text-slate-200 font-mono whitespace-pre-wrap">
Ready. Click Send to verify the proxy response stream.
        </div>
      </div>

    </div>

  </main>

  <!-- Toast Notification -->
  <div id="toast" class="fixed bottom-5 right-5 bg-indigo-600 text-white text-xs px-4 py-2.5 rounded-xl shadow-2xl transition-opacity duration-200 opacity-0 pointer-events-none z-50 flex items-center gap-2">
    <span>✓</span> <span id="toast-msg">Copied to clipboard</span>
  </div>

  <script>
    let allModels = [];
    let currentFilteredModels = [];
    let selectedModelId = 'anthropic/claude-3.7-sonnet';
    let currentTab = 'curl';

    // Initialize
    document.addEventListener('DOMContentLoaded', () => {
      checkHealth();
      fetchModels();
      setupEventListeners();
      updateOutputs();
    });

    async function checkHealth() {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        
        if (data.has_api_key) {
          dot.className = 'w-2 h-2 rounded-full bg-emerald-400';
          text.innerHTML = `Daemon Active • Key: <span class="font-mono text-indigo-400">${data.masked_key}</span>`;
        } else {
          dot.className = 'w-2 h-2 rounded-full bg-rose-500 animate-ping';
          text.innerHTML = `⚠️ No API Key in .env (Add OPENROUTER_API_KEY)`;
        }
      } catch (e) {
        document.getElementById('status-dot').className = 'w-2 h-2 rounded-full bg-rose-500';
        document.getElementById('status-text').textContent = 'Proxy Disconnected';
      }
    }

    async function fetchModels(forceRefresh = false) {
      const badge = document.getElementById('model-count-badge');
      badge.textContent = forceRefresh ? 'Refreshing...' : 'Fetching models...';
      
      try {
        const url = forceRefresh ? '/api/models?refresh=true' : '/api/models';
        const res = await fetch(url);
        const json = await res.json();
        allModels = json.data || [];
        badge.textContent = `${allModels.length} models available`;
        
        populateProviderFilter();
        renderModelsList(allModels);
        
        // If current selected model exists in list, sync card
        const match = allModels.find(m => m.id === selectedModelId);
        if (match) {
          updateSelectedModelCard(match);
        } else if (allModels.length > 0) {
          selectModel(allModels[0].id);
        }
      } catch (err) {
        badge.textContent = 'Failed to load models';
        console.error(err);
      }
    }

    function populateProviderFilter() {
      const select = document.getElementById('provider-filter');
      const providers = new Set();
      allModels.forEach(m => {
        const provider = m.id.split('/')[0];
        if (provider) providers.add(provider);
      });

      const sortedProviders = Array.from(providers).sort();
      select.innerHTML = '<option value="all">All Providers (' + sortedProviders.length + ')</option>';
      sortedProviders.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p.charAt(0).toUpperCase() + p.slice(1);
        select.appendChild(opt);
      });
    }

    function renderModelsList(models) {
      const container = document.getElementById('models-list-container');
      currentFilteredModels = models;
      if (models.length === 0) {
        container.innerHTML = '<div class="text-center py-6 text-slate-500 text-xs">No models match your search.</div>';
        return;
      }

      container.innerHTML = models.slice(0, 80).map(m => {
        const isSelected = m.id === selectedModelId;
        const ctxK = Math.round((m.context_length || 0) / 1000);
        const promptPrice = m.pricing ? (parseFloat(m.pricing.prompt) * 1000000).toFixed(2) : '0';
        const isReasoning = m.id.includes('r1') || m.id.includes('o1') || m.id.includes('o3') || m.id.includes('sonnet') || m.id.includes('deepseek-r1') || (m.supported_parameters && m.supported_parameters.includes('reasoning'));

        return `
          <div onclick="selectModel('${m.id}')" 
            class="p-2.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between text-xs ${
              isSelected 
                ? 'bg-indigo-600/20 border-indigo-500/80 text-white shadow-md' 
                : 'bg-dark-900/60 border-slate-800/80 text-slate-300 hover:border-slate-700 hover:bg-slate-800/40'
            }">
            <div class="min-w-0 pr-2">
              <div class="flex items-center gap-1.5">
                <span class="font-semibold truncate ${isSelected ? 'text-indigo-300' : 'text-slate-200'}">${m.id}</span>
                ${isReasoning ? '<span class="px-1 py-0.2 bg-violet-950 text-violet-300 border border-violet-800/50 rounded text-[9px]">🧠 CoT</span>' : ''}
              </div>
              <div class="text-[11px] text-slate-500 truncate">${m.name || m.id}</div>
            </div>
            <div class="text-right shrink-0 flex items-center gap-2">
              <span class="font-mono text-[10px] text-slate-400 bg-slate-800/80 px-1.5 py-0.5 rounded">${ctxK}k ctx</span>
              <span class="font-mono text-[10px] text-emerald-400">$${promptPrice}/M</span>
            </div>
          </div>
        `;
      }).join('');
    }

    function selectModel(modelId) {
      selectedModelId = modelId;
      const model = allModels.find(m => m.id === modelId) || { id: modelId, name: modelId, context_length: 8192 };
      updateSelectedModelCard(model);
      renderModelsList(currentFilteredModels);
      updateOutputs();
    }

    function updateSelectedModelCard(model) {
      document.getElementById('card-model-name').textContent = model.id;
      const ctxK = Math.round((model.context_length || 0) / 1000);
      document.getElementById('card-context-len').textContent = `${ctxK}k context`;
      document.getElementById('card-model-desc').textContent = model.description || model.name || 'OpenRouter model endpoint';
      
      if (model.pricing) {
        const pPrompt = (parseFloat(model.pricing.prompt || 0) * 1000000).toFixed(2);
        const pComp = (parseFloat(model.pricing.completion || 0) * 1000000).toFixed(2);
        document.getElementById('card-prompt-price').textContent = `$${pPrompt}/M`;
        document.getElementById('card-comp-price').textContent = `$${pComp}/M`;
      }
    }

    function setupEventListeners() {
      const searchInput = document.getElementById('model-search');
      const providerSelect = document.getElementById('provider-filter');
      
      const filterHandler = () => {
        const query = searchInput.value.toLowerCase().trim();
        const provider = providerSelect.value;

        const filtered = allModels.filter(m => {
          const matchQuery = !query || m.id.toLowerCase().includes(query) || (m.name && m.name.toLowerCase().includes(query)) || (m.description && m.description.toLowerCase().includes(query));
          const matchProvider = provider === 'all' || m.id.startsWith(provider + '/');
          return matchQuery && matchProvider;
        });
        renderModelsList(filtered);
      };

      searchInput.addEventListener('input', filterHandler);
      providerSelect.addEventListener('change', filterHandler);

      // Temperature slider
      const tempSlider = document.getElementById('param-temp');
      tempSlider.addEventListener('input', (e) => {
        document.getElementById('temp-val').textContent = e.target.value;
        updateOutputs();
      });

      // Top P slider
      const topPSlider = document.getElementById('param-topp');
      topPSlider.addEventListener('input', (e) => {
        document.getElementById('topp-val').textContent = e.target.value;
        updateOutputs();
      });

      // Max tokens
      const maxTokInput = document.getElementById('param-maxtok');
      maxTokInput.addEventListener('input', (e) => {
        document.getElementById('maxtok-val').textContent = e.target.value;
        updateOutputs();
      });

      // Reasoning
      document.getElementById('param-reasoning').addEventListener('change', updateOutputs);
      document.getElementById('param-system-prompt').addEventListener('input', updateOutputs);
      document.getElementById('param-provider-order').addEventListener('input', updateOutputs);
      document.getElementById('param-fallbacks').addEventListener('change', updateOutputs);

      // Keyboard shortcut ⌘K / Ctrl+K
      window.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
          e.preventDefault();
          searchInput.focus();
        }
      });
    }

    function applyFilterTag(tag) {
      if (tag === 'reasoning') {
        const filtered = allModels.filter(m => m.id.includes('r1') || m.id.includes('o1') || m.id.includes('o3') || m.id.includes('sonnet') || (m.supported_parameters && m.supported_parameters.includes('reasoning')));
        renderModelsList(filtered);
      } else if (tag === 'vision') {
        const filtered = allModels.filter(m => (m.architecture && m.architecture.modality && m.architecture.modality.includes('image')) || m.id.includes('vision') || m.id.includes('4o') || m.id.includes('gemini') || m.id.includes('sonnet'));
        renderModelsList(filtered);
      } else if (tag === 'free') {
        const filtered = allModels.filter(m => m.id.endsWith(':free') || (m.pricing && parseFloat(m.pricing.prompt) === 0));
        renderModelsList(filtered);
      } else if (tag === 'large-context') {
        const filtered = allModels.filter(m => (m.context_length || 0) >= 100000);
        renderModelsList(filtered);
      }
    }

    function clearFilterTags() {
      document.getElementById('model-search').value = '';
      document.getElementById('provider-filter').value = 'all';
      renderModelsList(allModels);
    }

    function getCurrentParams() {
      const temp = parseFloat(document.getElementById('param-temp').value);
      const topP = parseFloat(document.getElementById('param-topp').value);
      const maxTok = parseInt(document.getElementById('param-maxtok').value, 10);
      const reasoning = document.getElementById('param-reasoning').value;
      const sysPrompt = document.getElementById('param-system-prompt').value.trim();
      const providerOrder = document.getElementById('param-provider-order').value.trim();
      const fallbacks = document.getElementById('param-fallbacks').checked;

      return {
        model: selectedModelId,
        temperature: temp,
        top_p: topP,
        max_tokens: maxTok,
        reasoning_effort: reasoning !== 'none' ? reasoning : undefined,
        system_prompt: sysPrompt || undefined,
        provider_order: providerOrder || undefined,
        allow_fallbacks: !fallbacks ? false : undefined
      };
    }

    function updateOutputs() {
      const origin = window.location.origin;
      const p = getCurrentParams();

      // 1. Path-based URL: /p/provider:model_name/v1
      // Replace '/' with ':' in model name for cleanest URL pathing
      const modelSlug = p.model.replace('/', ':');
      const pathBaseUrl = `${origin}/p/${modelSlug}/v1`;
      document.getElementById('path-base-url').textContent = pathBaseUrl;

      // 2. Query string URL: /v1?model=...&temp=...
      const queryParts = [`model=${encodeURIComponent(p.model)}`];
      if (p.temperature !== 0.7) queryParts.push(`temperature=${p.temperature}`);
      if (p.top_p !== 1.0) queryParts.push(`top_p=${p.top_p}`);
      if (p.max_tokens !== 4096) queryParts.push(`max_tokens=${p.max_tokens}`);
      if (p.reasoning_effort) queryParts.push(`reasoning_effort=${p.reasoning_effort}`);
      if (p.provider_order) queryParts.push(`provider_order=${encodeURIComponent(p.provider_order)}`);
      
      const queryBaseUrl = `${origin}/v1?${queryParts.join('&')}`;
      document.getElementById('query-base-url').textContent = queryBaseUrl;

      // Direct completions endpoint
      const completionsUrl = `${origin}/p/${modelSlug}/v1/chat/completions`;
      document.getElementById('completions-endpoint-url').textContent = completionsUrl;

      renderSnippet();
    }

    function setSnippetTab(tab) {
      currentTab = tab;
      ['curl', 'python', 'js', 'aider'].forEach(t => {
        const el = document.getElementById(`tab-${t}`);
        if (t === tab) {
          el.className = 'font-semibold text-indigo-400 border-b-2 border-indigo-500 pb-1';
        } else {
          el.className = 'text-slate-400 hover:text-slate-200 pb-1';
        }
      });
      renderSnippet();
    }

    function renderSnippet() {
      const origin = window.location.origin;
      const p = getCurrentParams();
      const modelSlug = p.model.replace('/', ':');
      const box = document.getElementById('snippet-box');

      if (currentTab === 'curl') {
        box.textContent = `curl ${origin}/p/${modelSlug}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer sk-proxy-local" \\
  -d '{
    "messages": [{"role": "user", "content": "Hello via OpenRouter proxy!"}],
    "temperature": ${p.temperature},
    "max_tokens": ${p.max_tokens},
    "stream": true
  }'`;
      } else if (currentTab === 'python') {
        box.textContent = `from openai import OpenAI

client = OpenAI(
    base_url="${origin}/p/${modelSlug}/v1",
    api_key="sk-proxy-local"  # Server uses OPENROUTER_API_KEY
)

response = client.chat.completions.create(
    model="${p.model}",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=${p.temperature},
    max_tokens=${p.max_tokens},
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)`;
      } else if (currentTab === 'js') {
        box.textContent = `import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "${origin}/p/${modelSlug}/v1",
  apiKey: "sk-proxy-local"
});

const stream = await openai.chat.completions.create({
  model: "${p.model}",
  messages: [{ role: "user", content: "Hello!" }],
  stream: true
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}`;
      } else if (currentTab === 'aider') {
        box.textContent = `# Set environment for Aider / Cursor / Roo Code
export OPENAI_API_BASE="${origin}/p/${modelSlug}/v1"
export OPENAI_API_KEY="sk-proxy-local"

# Run aider with this configured proxy
aider --openai-api-base="${origin}/p/${modelSlug}/v1"`;
      }
    }

    async function sendTestPrompt() {
      const btn = document.getElementById('test-send-btn');
      const input = document.getElementById('test-input');
      const output = document.getElementById('test-output');
      const latency = document.getElementById('test-latency');
      const p = getCurrentParams();
      const prompt = input.value.trim();

      if (!prompt) return;

      btn.disabled = true;
      btn.classList.add('opacity-50');
      document.getElementById('test-btn-icon').textContent = '⏳';
      output.textContent = 'Connecting and streaming response...\\n';
      
      const startTime = performance.now();
      const modelSlug = p.model.replace('/', ':');
      const endpoint = `/p/${modelSlug}/v1/chat/completions`;

      try {
        const bodyPayload = {
          messages: [{ role: 'user', content: prompt }],
          temperature: p.temperature,
          top_p: p.top_p,
          max_tokens: p.max_tokens,
          stream: true
        };
        if (p.reasoning_effort) bodyPayload.reasoning_effort = p.reasoning_effort;
        if (p.system_prompt) bodyPayload.messages.unshift({ role: 'system', content: p.system_prompt });

        const res = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer sk-proxy-local'
          },
          body: JSON.stringify(bodyPayload)
        });

        if (!res.ok) {
          const errText = await res.text();
          output.textContent = `❌ Error HTTP ${res.status}:\\n${errText}`;
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        output.textContent = '';
        let fullText = '';
        let firstTokenTime = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          if (!firstTokenTime) {
            firstTokenTime = ((performance.now() - startTime) / 1000).toFixed(2);
            latency.textContent = `TTFT: ${firstTokenTime}s`;
          }

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.replace('data: ', '').trim();
              if (dataStr === '[DONE]') continue;
              try {
                const parsed = JSON.parse(dataStr);
                const delta = parsed.choices?.[0]?.delta?.content || '';
                const reasoning = parsed.choices?.[0]?.delta?.reasoning || parsed.choices?.[0]?.delta?.reasoning_content || '';
                
                if (reasoning) {
                  fullText += `[Thinking: ${reasoning}]`;
                }
                if (delta) {
                  fullText += delta;
                }
                output.textContent = fullText;
                output.scrollTop = output.scrollHeight;
              } catch (e) {
                // partial line
              }
            }
          }
        }

        const totalTime = ((performance.now() - startTime) / 1000).toFixed(2);
        latency.textContent = `Done in ${totalTime}s (TTFT: ${firstTokenTime}s)`;

      } catch (e) {
        output.textContent = `❌ Fetch Error: ${e.message}`;
      } finally {
        btn.disabled = false;
        btn.classList.remove('opacity-50');
        document.getElementById('test-btn-icon').textContent = '🚀';
      }
    }

    function showToast(msg) {
      const toast = document.getElementById('toast');
      document.getElementById('toast-msg').textContent = msg;
      toast.classList.remove('opacity-0');
      toast.classList.add('opacity-100');
      setTimeout(() => {
        toast.classList.remove('opacity-100');
        toast.classList.add('opacity-0');
      }, 2000);
    }

    function copyToClipboard(elementId) {
      const text = document.getElementById(elementId).textContent.trim();
      navigator.clipboard.writeText(text);
      showToast('Copied endpoint URL to clipboard!');
    }

    function copySnippet() {
      const text = document.getElementById('snippet-box').textContent.trim();
      navigator.clipboard.writeText(text);
      showToast('Copied code snippet!');
    }
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


def main():
    port = int(os.getenv("PORT", "18080"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"🚀 Starting OpenRouter LiteLLM Proxy on http://{host}:{port}")
    print(f"🌐 Web UI dashboard available at http://localhost:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
