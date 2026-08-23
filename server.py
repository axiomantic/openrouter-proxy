#!/usr/bin/env python3
"""
OpenRouter LiteLLM Proxy Service
Provides OpenAI-compatible endpoints with dynamic URL-based configuration
and a clean, minimalist web UI for model selection and URL generation.
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

    # Merge configuration: URL config overrides body defaults if specified
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

    # Reasoning effort support
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
@app.post("/p/{config_path:path}/v1/chat/completions")
async def path_chat_completions(request: Request, config_path: str):
    url_config = extract_url_config(request, path_param=config_path)
    return await handle_proxy_completion(request, url_config)


# 3. Base64 configuration route: /cfg/{b64_config}/v1/chat/completions
@app.post("/cfg/{b64_config}/v1/chat/completions")
async def b64_chat_completions(request: Request, b64_config: str):
    url_config = extract_url_config(request, path_param=f"b64:{b64_config}")
    return await handle_proxy_completion(request, url_config)


# 4. Catch-all for alternate roots
@app.post("/chat/completions")
@app.post("/p/{config_path:path}/chat/completions")
async def alternate_chat_completions(request: Request, config_path: Optional[str] = None):
    url_config = extract_url_config(request, path_param=config_path)
    return await handle_proxy_completion(request, url_config)


# ==============================================================================
# Web UI Dashboard (Minimalist & Undecorated)
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
@app.head("/")
@app.head("/ui")
async def web_dashboard(request: Request):
    """Minimalist, undecorated configuration generator and tester."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>openrouter-proxy</title>
  <style>
    :root {
      --bg: #09090b;
      --surface: #121215;
      --surface-hover: #18181c;
      --border: #27272a;
      --border-focus: #52525b;
      --text: #f4f4f5;
      --text-muted: #71717a;
      --text-dim: #52525b;
      --accent: #e4e4e7;
      --accent-bg: #27272a;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 13px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      padding: 24px;
    }
    a { color: inherit; text-decoration: none; }
    input, select, textarea, button {
      font-family: inherit;
      font-size: 12px;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 4px;
      padding: 6px 10px;
      outline: none;
    }
    input:focus, select:focus, textarea:focus { border-color: var(--border-focus); }
    button {
      cursor: pointer;
      background: var(--surface);
      border: 1px solid var(--border);
      transition: background 0.15s, border-color 0.15s;
    }
    button:hover { background: var(--surface-hover); border-color: var(--border-focus); }
    .mono { font-family: var(--font-mono); }
    .container { max-width: 1100px; margin: 0 auto; display: flex; flex-direction: column; gap: 24px; }
    
    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    .brand { display: flex; align-items: baseline; gap: 12px; }
    .brand h1 { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
    .brand span { font-size: 12px; color: var(--text-muted); }
    .status {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }
    .dot { width: 6px; height: 6px; border-radius: 50%; background: #71717a; }
    .dot.active { background: #22c55e; }
    .dot.warn { background: #eab308; }

    /* Layout */
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
    
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .panel-title {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      font-weight: 600;
    }

    /* Model list */
    .model-filters { display: flex; gap: 8px; }
    .model-filters input { flex: 1; }
    .model-list {
      max-height: 220px;
      overflow-y: auto;
      border: 1px solid var(--border);
      border-radius: 4px;
      background: var(--bg);
    }
    .model-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 10px;
      border-bottom: 1px solid #1c1c20;
      cursor: pointer;
      font-size: 12px;
    }
    .model-row:last-child { border-bottom: none; }
    .model-row:hover { background: var(--surface-hover); }
    .model-row.selected {
      background: var(--accent-bg);
      color: #fff;
    }
    .model-id { font-family: var(--font-mono); font-size: 11px; }
    .model-meta { display: flex; gap: 8px; color: var(--text-muted); font-size: 11px; font-family: var(--font-mono); }
    .model-row.selected .model-meta { color: #d4d4d8; }

    /* Parameters */
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .form-group { display: flex; flex-direction: column; gap: 4px; }
    .form-group label {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: var(--text-muted);
    }
    .form-group label span { font-family: var(--font-mono); color: var(--text); }
    .form-group input[type="range"] {
      padding: 0;
      background: transparent;
      border: none;
      height: 20px;
      accent-color: var(--accent);
    }

    /* Endpoints */
    .endpoint-block { display: flex; flex-direction: column; gap: 6px; }
    .endpoint-label { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); }
    .endpoint-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 8px 10px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text);
      word-break: break-all;
      user-select: all;
    }

    /* Code Snippets */
    .tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
    .tab {
      background: none;
      border: none;
      color: var(--text-muted);
      padding: 2px 8px;
      font-size: 11px;
      border-radius: 3px;
    }
    .tab.active { background: var(--accent-bg); color: var(--text); }
    pre {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 10px;
      font-family: var(--font-mono);
      font-size: 11px;
      overflow-x: auto;
      white-space: pre;
      line-height: 1.4;
    }

    /* Test prompt */
    .test-form { display: flex; gap: 8px; }
    .test-form input { flex: 1; }
    .test-output {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 10px;
      font-family: var(--font-mono);
      font-size: 11px;
      min-height: 80px;
      max-height: 160px;
      overflow-y: auto;
      white-space: pre-wrap;
      color: var(--text);
    }

    /* Toast */
    #toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: var(--text);
      color: var(--bg);
      padding: 6px 12px;
      border-radius: 4px;
      font-size: 11px;
      font-family: var(--font-mono);
      opacity: 0;
      transition: opacity 0.15s;
      pointer-events: none;
    }
  </style>
</head>
<body>

  <div class="container">
    <!-- Header -->
    <header>
      <div class="brand">
        <h1>openrouter-proxy</h1>
        <span>v1.0.0</span>
      </div>
      <div class="status">
        <span class="dot" id="status-dot"></span>
        <span id="status-text">connecting</span>
      </div>
    </header>

    <!-- Main Grid -->
    <div class="grid">
      
      <!-- Left: Model & Parameters -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        
        <!-- Model Selection -->
        <div class="panel">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="panel-title">Model</div>
            <div id="model-count" style="font-size: 11px; color: var(--text-dim); font-family: var(--font-mono);">0 models</div>
          </div>

          <div class="model-filters">
            <input type="text" id="model-search" placeholder="Filter models (e.g. claude, gpt-4o, deepseek)..." autocomplete="off">
            <select id="provider-filter">
              <option value="all">All</option>
            </select>
          </div>

          <div class="model-list" id="models-list">
            <div style="padding: 20px; text-align: center; color: var(--text-dim); font-size: 11px;">Loading models...</div>
          </div>

          <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">
            <span>Selected: <span id="selected-model-name" style="color: var(--text);">anthropic/claude-3.7-sonnet</span></span>
            <span id="selected-model-ctx">200k ctx</span>
          </div>
        </div>

        <!-- Parameters -->
        <div class="panel">
          <div class="panel-title">Parameters</div>
          
          <div class="form-grid">
            <div class="form-group">
              <label>Temperature <span id="val-temp">0.7</span></label>
              <input type="range" id="param-temp" min="0" max="2" step="0.05" value="0.7">
            </div>
            <div class="form-group">
              <label>Top P <span id="val-topp">1.0</span></label>
              <input type="range" id="param-topp" min="0" max="1" step="0.05" value="1.0">
            </div>
            <div class="form-group">
              <label>Max Output Tokens</label>
              <input type="number" id="param-maxtok" min="1" max="128000" step="256" value="4096">
            </div>
            <div class="form-group">
              <label>Reasoning Effort</label>
              <select id="param-reasoning">
                <option value="none">Default / None</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>

          <div class="form-group" style="margin-top: 4px;">
            <label>System Prompt Override</label>
            <input type="text" id="param-sysprompt" placeholder="Optional system instruction...">
          </div>

          <div class="form-group">
            <label>Provider Order</label>
            <input type="text" id="param-providers" placeholder="e.g. Together,Fireworks,DeepInfra">
          </div>
        </div>

      </div>

      <!-- Right: Endpoints & Test -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        
        <!-- Generated URLs -->
        <div class="panel">
          <div class="panel-title">Configured Endpoints</div>

          <div class="endpoint-block">
            <div class="endpoint-label">
              <span>Path Base URL (OpenAI SDK / Cursor / Aider)</span>
              <button onclick="copyText('url-path')">Copy</button>
            </div>
            <div class="endpoint-box" id="url-path">http://localhost:18080/p/anthropic:claude-3.7-sonnet/v1</div>
          </div>

          <div class="endpoint-block">
            <div class="endpoint-label">
              <span>Query String Base URL</span>
              <button onclick="copyText('url-query')">Copy</button>
            </div>
            <div class="endpoint-box" id="url-query">http://localhost:18080/v1?model=anthropic/claude-3.7-sonnet&temperature=0.7</div>
          </div>

          <!-- Code Snippets -->
          <div style="margin-top: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <div class="tabs">
                <button class="tab active" onclick="setTab('curl')">cURL</button>
                <button class="tab" onclick="setTab('python')">Python</button>
                <button class="tab" onclick="setTab('js')">Node.js</button>
                <button class="tab" onclick="setTab('env')">Env Vars</button>
              </div>
              <button onclick="copySnippet()" style="font-size: 11px;">Copy Code</button>
            </div>
            <pre id="snippet-code"></pre>
          </div>
        </div>

        <!-- Live Endpoint Test -->
        <div class="panel">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="panel-title">Test Endpoint</div>
            <div id="test-latency" style="font-size: 11px; color: var(--text-dim); font-family: var(--font-mono);"></div>
          </div>

          <div class="test-form">
            <input type="text" id="test-input" value="Write a one-sentence summary of quantum computing.">
            <button id="test-btn" onclick="runTest()">Send</button>
          </div>

          <div class="test-output" id="test-output">Ready.</div>
        </div>

      </div>

    </div>
  </div>

  <div id="toast">Copied</div>

  <script>
    let allModels = [];
    let selectedModel = 'anthropic/claude-3.7-sonnet';
    let currentTab = 'curl';

    document.addEventListener('DOMContentLoaded', () => {
      checkHealth();
      fetchModels();
      initEvents();
      update();
    });

    async function checkHealth() {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        const dot = document.getElementById('status-dot');
        const txt = document.getElementById('status-text');
        if (data.has_api_key) {
          dot.className = 'dot active';
          txt.textContent = `online • ${data.masked_key}`;
        } else {
          dot.className = 'dot warn';
          txt.textContent = 'online • no api key in .env';
        }
      } catch (e) {
        document.getElementById('status-dot').className = 'dot';
        document.getElementById('status-text').textContent = 'offline';
      }
    }

    async function fetchModels() {
      try {
        const res = await fetch('/api/models');
        const json = await res.json();
        allModels = json.data || [];
        document.getElementById('model-count').textContent = `${allModels.length} models`;
        buildProviders();
        renderModels(allModels);
      } catch (e) {
        document.getElementById('models-list').innerHTML = '<div style="padding: 10px; color: var(--text-dim);">Failed to load models.</div>';
      }
    }

    function buildProviders() {
      const select = document.getElementById('provider-filter');
      const set = new Set();
      allModels.forEach(m => {
        const p = m.id.split('/')[0];
        if (p) set.add(p);
      });
      Array.from(set).sort().forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p;
        select.appendChild(opt);
      });
    }

    function renderModels(list) {
      const container = document.getElementById('models-list');
      if (!list.length) {
        container.innerHTML = '<div style="padding: 10px; color: var(--text-dim);">No matching models</div>';
        return;
      }
      container.innerHTML = list.slice(0, 100).map(m => {
        const isSel = m.id === selectedModel;
        const ctx = Math.round((m.context_length || 0) / 1000);
        return `
          <div class="model-row ${isSel ? 'selected' : ''}" onclick="selectModel('${m.id}')">
            <span class="model-id">${m.id}</span>
            <div class="model-meta">
              <span>${ctx}k</span>
            </div>
          </div>
        `;
      }).join('');
    }

    function selectModel(id) {
      selectedModel = id;
      document.getElementById('selected-model-name').textContent = id;
      const m = allModels.find(x => x.id === id);
      if (m) {
        const ctx = Math.round((m.context_length || 0) / 1000);
        document.getElementById('selected-model-ctx').textContent = `${ctx}k ctx`;
      }
      renderModels(getFilteredModels());
      update();
    }

    function getFilteredModels() {
      const q = document.getElementById('model-search').value.toLowerCase().trim();
      const p = document.getElementById('provider-filter').value;
      return allModels.filter(m => {
        const matchesQ = !q || m.id.toLowerCase().includes(q) || (m.name && m.name.toLowerCase().includes(q));
        const matchesP = p === 'all' || m.id.startsWith(p + '/');
        return matchesQ && matchesP;
      });
    }

    function initEvents() {
      document.getElementById('model-search').addEventListener('input', () => renderModels(getFilteredModels()));
      document.getElementById('provider-filter').addEventListener('change', () => renderModels(getFilteredModels()));

      document.getElementById('param-temp').addEventListener('input', (e) => {
        document.getElementById('val-temp').textContent = e.target.value;
        update();
      });
      document.getElementById('param-topp').addEventListener('input', (e) => {
        document.getElementById('val-topp').textContent = e.target.value;
        update();
      });
      document.getElementById('param-maxtok').addEventListener('input', update);
      document.getElementById('param-reasoning').addEventListener('change', update);
      document.getElementById('param-sysprompt').addEventListener('input', update);
      document.getElementById('param-providers').addEventListener('input', update);
    }

    function getParams() {
      return {
        model: selectedModel,
        temp: parseFloat(document.getElementById('param-temp').value),
        topP: parseFloat(document.getElementById('param-topp').value),
        maxTok: parseInt(document.getElementById('param-maxtok').value, 10),
        reasoning: document.getElementById('param-reasoning').value,
        sysPrompt: document.getElementById('param-sysprompt').value.trim(),
        providers: document.getElementById('param-providers').value.trim()
      };
    }

    function update() {
      const origin = window.location.origin;
      const p = getParams();
      const slug = p.model.replace('/', ':');

      // Path URL
      const pathUrl = `${origin}/p/${slug}/v1`;
      document.getElementById('url-path').textContent = pathUrl;

      // Query URL
      const q = [`model=${encodeURIComponent(p.model)}`];
      if (p.temp !== 0.7) q.push(`temperature=${p.temp}`);
      if (p.topP !== 1.0) q.push(`top_p=${p.topP}`);
      if (p.maxTok !== 4096) q.push(`max_tokens=${p.maxTok}`);
      if (p.reasoning !== 'none') q.push(`reasoning_effort=${p.reasoning}`);
      if (p.providers) q.push(`provider_order=${encodeURIComponent(p.providers)}`);

      document.getElementById('url-query').textContent = `${origin}/v1?${q.join('&')}`;

      renderSnippet();
    }

    function setTab(tab) {
      currentTab = tab;
      document.querySelectorAll('.tab').forEach(el => {
        el.classList.toggle('active', el.textContent.toLowerCase().includes(tab));
      });
      renderSnippet();
    }

    function renderSnippet() {
      const origin = window.location.origin;
      const p = getParams();
      const slug = p.model.replace('/', ':');
      const box = document.getElementById('snippet-code');

      if (currentTab === 'curl') {
        box.textContent = `curl ${origin}/p/${slug}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer sk-proxy-local" \\
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": ${p.temp},
    "max_tokens": ${p.maxTok},
    "stream": true
  }'`;
      } else if (currentTab === 'python') {
        box.textContent = `from openai import OpenAI

client = OpenAI(
    base_url="${origin}/p/${slug}/v1",
    api_key="sk-proxy-local"
)

response = client.chat.completions.create(
    model="${p.model}",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=${p.temp},
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`;
      } else if (currentTab === 'js') {
        box.textContent = `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "${origin}/p/${slug}/v1",
  apiKey: "sk-proxy-local"
});

const stream = await client.chat.completions.create({
  model: "${p.model}",
  messages: [{ role: "user", content: "Hello" }],
  stream: true
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}`;
      } else if (currentTab === 'env') {
        box.textContent = `export OPENAI_BASE_URL="${origin}/p/${slug}/v1"
export OPENAI_API_KEY="sk-proxy-local"`;
      }
    }

    async function runTest() {
      const btn = document.getElementById('test-btn');
      const input = document.getElementById('test-input');
      const out = document.getElementById('test-output');
      const lat = document.getElementById('test-latency');
      const p = getParams();
      const text = input.value.trim();
      if (!text) return;

      btn.disabled = true;
      btn.textContent = '...';
      out.textContent = '';
      lat.textContent = '';
      
      const start = performance.now();
      const slug = p.model.replace('/', ':');

      try {
        const payload = {
          messages: [{ role: 'user', content: text }],
          temperature: p.temp,
          top_p: p.topP,
          max_tokens: p.maxTok,
          stream: true
        };
        if (p.reasoning !== 'none') payload.reasoning_effort = p.reasoning;
        if (p.sysPrompt) payload.messages.unshift({ role: 'system', content: p.sysPrompt });

        const res = await fetch(`/p/${slug}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer sk-proxy-local' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          out.textContent = `HTTP ${res.status}: ${await res.text()}`;
          return;
        }

        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let full = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = dec.decode(value, { stream: true });
          const lines = chunk.split('\\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.replace('data: ', '').trim();
              if (dataStr === '[DONE]') continue;
              try {
                const parsed = JSON.parse(dataStr);
                const delta = parsed.choices?.[0]?.delta?.content || '';
                full += delta;
                out.textContent = full;
              } catch (e) {}
            }
          }
        }
        const took = ((performance.now() - start) / 1000).toFixed(2);
        lat.textContent = `${took}s`;
      } catch (e) {
        out.textContent = `Error: ${e.message}`;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Send';
      }
    }

    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.textContent = msg;
      toast.style.opacity = '1';
      setTimeout(() => { toast.style.opacity = '0'; }, 1500);
    }

    function copyText(id) {
      navigator.clipboard.writeText(document.getElementById(id).textContent.trim());
      showToast('Copied URL');
    }

    function copySnippet() {
      navigator.clipboard.writeText(document.getElementById('snippet-code').textContent.trim());
      showToast('Copied snippet');
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
