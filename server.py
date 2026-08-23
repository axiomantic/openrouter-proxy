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


def compute_model_popularity(model: Dict[str, Any]) -> int:
    """Calculates popularity score for sorting models descending."""
    mid = model.get("id", "").lower()
    score = 0

    # Flagship model tiers
    if "claude-3.7-sonnet" in mid or "claude-3-7-sonnet" in mid or "claude-sonnet-5" in mid:
        score += 100000
    elif "deepseek-r1" in mid:
        score += 95000
    elif "claude-3.5-sonnet" in mid or "claude-3-5-sonnet" in mid or "claude-opus-5" in mid:
        score += 90000
    elif "gpt-4o" in mid and "mini" not in mid and "batch" not in mid:
        score += 85000
    elif "o3-mini" in mid or "o1" in mid:
        score += 80000
    elif "gemini-2.5-pro" in mid or "gemini-2.0-flash" in mid:
        score += 75000
    elif "llama-3.3-70b" in mid:
        score += 70000
    elif "deepseek-chat" in mid or "deepseek-v3" in mid:
        score += 65000
    elif "gpt-4o-mini" in mid and "batch" not in mid:
        score += 60000
    elif "qwen-2.5-coder-32b" in mid or "qwq" in mid:
        score += 55000
    elif "mistral-large" in mid or "codestral" in mid:
        score += 50000

    # Provider weights
    if mid.startswith("anthropic/"):
        score += 30000
    elif mid.startswith("openai/"):
        score += 28000
    elif mid.startswith("deepseek/"):
        score += 26000
    elif mid.startswith("google/"):
        score += 24000
    elif mid.startswith("meta-llama/"):
        score += 22000
    elif mid.startswith("qwen/"):
        score += 20000
    elif mid.startswith("mistralai/"):
        score += 18000
    elif mid.startswith("x-ai/"):
        score += 16000
    elif mid.startswith("cohere/"):
        score += 12000

    # Capability bonuses
    if "r1" in mid or "o1" in mid or "o3" in mid:
        score += 3000
    if "sonnet" in mid or "opus" in mid or "4o" in mid or "70b" in mid or "405b" in mid or "coder" in mid:
        score += 2000
    if model.get("context_length", 0) >= 128000:
        score += 1000

    # Batch / experimental penalties
    if ":batch" in mid:
        score -= 5000
    if ":nitro" in mid or ":floor" in mid:
        score -= 1000

    return score


async def fetch_openrouter_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch, cache, and sort models by popularity descending from OpenRouter."""
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
                # Sort by popularity descending
                models_data.sort(key=compute_model_popularity, reverse=True)
                MODELS_CACHE["data"] = models_data
                MODELS_CACHE["timestamp"] = now
                logger.info(f"Loaded and sorted {len(models_data)} models by popularity from OpenRouter")
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


# Cache for model endpoints/providers
MODEL_ENDPOINTS_CACHE: Dict[str, Any] = {}

async def fetch_model_endpoints(model_id: str, api_key: Optional[str] = None) -> List[str]:
    """Fetch supported provider names for a specific model."""
    if model_id in MODEL_ENDPOINTS_CACHE:
        return MODEL_ENDPOINTS_CACHE[model_id]

    headers = {}
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{OPENROUTER_BASE_URL}/models/{model_id}/endpoints", headers=headers)
            if resp.status_code == 200:
                endpoints_data = resp.json().get("data", {}).get("endpoints", [])
                providers = []
                for ep in endpoints_data:
                    p_name = ep.get("provider_name")
                    if not p_name and ep.get("name"):
                        p_name = ep.get("name").split("|")[0].strip()
                    if p_name and p_name not in providers:
                        providers.append(p_name)
                
                # If no sub-providers listed (e.g. Anthropic native), use the author/org
                if not providers and "/" in model_id:
                    author = model_id.split("/")[0].capitalize()
                    providers.append(author)

                MODEL_ENDPOINTS_CACHE[model_id] = providers
                return providers
    except Exception as e:
        logger.error(f"Error fetching endpoints for {model_id}: {e}")

    # Fallback to model author
    fallback = [model_id.split("/")[0].capitalize()] if "/" in model_id else []
    return fallback


@app.get("/api/models/{model_author}/{model_slug}/endpoints")
@app.get("/api/models/{model_slug}/endpoints")
async def get_model_endpoints(model_slug: str, model_author: Optional[str] = None, request: Request = None):
    """Retrieve actual supported providers for a specific model."""
    model_id = f"{model_author}/{model_slug}" if model_author else model_slug
    providers = await fetch_model_endpoints(model_id, get_api_key(request) if request else None)
    return {"model": model_id, "providers": providers}


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
    """Large-type, model-reactive, unboxed configuration generator."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenRouter Proxy</title>
  <style>
    :root {
      --bg: #0c0c0e;
      --input-bg: #16161a;
      --item-bg: #1e1e24;
      --border: #2c2c34;
      --border-focus: #71717a;
      --text: #f4f4f6;
      --text-muted: #a1a1aa;
      --text-subtle: #71717a;
      --accent: #ffffff;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 16px;
      line-height: 1.6;
      padding: 48px 24px;
      -webkit-font-smoothing: antialiased;
    }
    .wrap {
      max-width: 780px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 40px;
    }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }
    h1 { font-size: 22px; font-weight: 600; letter-spacing: -0.02em; }
    .status {
      font-size: 14px;
      font-family: var(--mono);
      color: var(--text-muted);
    }
    .status.online { color: #4ade80; }
    .status.warn { color: #facc15; }

    /* Section */
    section {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    h2 {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-subtle);
      font-weight: 600;
    }

    /* Inputs */
    label {
      font-size: 15px;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      margin-bottom: 6px;
    }
    label span {
      font-family: var(--mono);
      color: var(--text);
      font-weight: 500;
    }
    .param-note {
      font-size: 13px;
      font-family: var(--sans);
      font-weight: 400;
    }
    .note-disabled { color: var(--text-subtle); }
    .note-supported { color: #4ade80; }
    .note-reasoning { color: #a78bfa; }

    input[type="text"], input[type="number"], select {
      width: 100%;
      background: var(--input-bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: inherit;
      font-size: 15px;
      padding: 10px 14px;
      border-radius: 6px;
      outline: none;
      transition: border-color 0.15s, opacity 0.15s;
    }
    input:disabled, select:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      border-color: #222228;
    }
    input[type="text"]:focus, input[type="number"]:focus, select:focus {
      border-color: var(--border-focus);
    }
    input[type="range"] {
      width: 100%;
      height: 24px;
      accent-color: var(--text);
      background: transparent;
      cursor: pointer;
    }
    input[type="range"]:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    /* Model select list */
    .model-selector {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .model-list-box {
      max-height: 200px;
      overflow-y: auto;
      background: var(--input-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
    }
    .model-option {
      padding: 10px 14px;
      font-family: var(--mono);
      font-size: 14px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #1f1f26;
    }
    .model-option:last-child { border-bottom: none; }
    .model-option:hover { background: #22222a; }
    .model-option.active { background: #2a2a36; color: #ffffff; font-weight: 600; }
    .model-ctx { color: var(--text-subtle); font-size: 13px; }

    /* Parameters grid */
    .params-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 600px) { .params-grid { grid-template-columns: 1fr; } }

    /* Draggable Provider Order */
    .provider-container {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .provider-add-row {
      display: flex;
      gap: 8px;
    }
    .provider-add-row select {
      flex: 1;
    }
    .provider-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-height: 10px;
    }
    .provider-item {
      background: var(--item-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: grab;
      user-select: none;
      transition: background 0.15s, opacity 0.15s;
    }
    .provider-item:active { cursor: grabbing; }
    .provider-item.dragging { opacity: 0.4; background: #2a2a34; }
    .provider-drag-handle {
      color: var(--text-subtle);
      font-family: var(--mono);
      font-size: 14px;
      margin-right: 8px;
    }
    .provider-actions {
      display: flex;
      gap: 4px;
    }
    .btn-small {
      padding: 2px 8px;
      font-size: 12px;
      border-radius: 4px;
      background: var(--input-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
    }
    .btn-small:hover { color: var(--text); background: #262630; }

    /* Output Endpoint Box */
    .endpoint-item {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .endpoint-row {
      display: flex;
      align-items: stretch;
      gap: 8px;
    }
    .code-box {
      flex: 1;
      background: var(--input-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 14px;
      font-family: var(--mono);
      font-size: 14px;
      color: var(--text);
      word-break: break-all;
      user-select: all;
    }
    button {
      background: var(--input-bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 14px;
      font-weight: 500;
      padding: 0 18px;
      border-radius: 6px;
      cursor: pointer;
      white-space: nowrap;
      transition: background 0.15s, border-color 0.15s;
    }
    button:hover {
      background: #22222a;
      border-color: var(--border-focus);
    }
    button:active { transform: scale(0.99); }

    /* Test prompt */
    .test-box {
      display: flex;
      gap: 8px;
    }
    .test-output {
      background: var(--input-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
      font-family: var(--mono);
      font-size: 14px;
      line-height: 1.5;
      min-height: 90px;
      max-height: 240px;
      overflow-y: auto;
      white-space: pre-wrap;
    }

    /* Toast */
    #toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #f4f4f6;
      color: #0c0c0e;
      font-size: 14px;
      font-weight: 500;
      padding: 10px 18px;
      border-radius: 6px;
      opacity: 0;
      transition: opacity 0.15s;
      pointer-events: none;
    }
  </style>
</head>
<body>

  <div class="wrap">
    
    <!-- Top Header -->
    <header>
      <h1>OpenRouter Proxy</h1>
      <div class="status" id="status-indicator">connecting...</div>
    </header>

    <!-- 1. Model Selection -->
    <section>
      <h2>1. Choose Model</h2>
      <div class="model-selector">
        <input type="text" id="model-search" placeholder="Filter models by name or provider (e.g. claude, gpt-4o, deepseek)..." autocomplete="off">
        <div class="model-list-box" id="models-list">
          <div style="padding: 14px; color: var(--text-subtle);">Loading models...</div>
        </div>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 14px; color: var(--text-muted); font-family: var(--mono);">
        <span>Selected: <strong id="selected-model-text" style="color: var(--text);">anthropic/claude-3.7-sonnet</strong></span>
        <span id="selected-model-info">200k context</span>
      </div>
    </section>

    <!-- 2. Dynamic Model Parameters -->
    <section>
      <h2>2. Model Parameters</h2>

      <div class="params-grid">
        <!-- Temperature -->
        <div id="wrapper-temp">
          <label>
            <span>Temperature</span>
            <span id="lbl-temp-note" class="param-note"><span id="lbl-temp">0.7</span></span>
          </label>
          <input type="range" id="input-temp" min="0" max="2" step="0.05" value="0.7">
        </div>

        <!-- Top P -->
        <div id="wrapper-topp">
          <label>
            <span>Top P</span>
            <span id="lbl-topp-note" class="param-note"><span id="lbl-topp">1.0</span></span>
          </label>
          <input type="range" id="input-topp" min="0" max="1" step="0.05" value="1.0">
        </div>

        <!-- Max Output Tokens (Jumps automatically to model limit) -->
        <div id="wrapper-maxtok">
          <label>
            <span>Max Output Tokens</span>
            <span id="lbl-maxtok-limit" class="param-note note-supported">(max: 8,192)</span>
          </label>
          <input type="number" id="input-maxtok" min="1" max="8192" step="256" value="8192">
        </div>

        <!-- Reasoning Effort (Disabled if not supported) -->
        <div id="wrapper-reasoning">
          <label>
            <span>Reasoning Effort</span>
            <span id="lbl-reasoning-note" class="param-note note-reasoning">Supported</span>
          </label>
          <select id="input-reasoning">
            <option value="none">Default (None)</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High (Deep Reason)</option>
          </select>
        </div>
      </div>

      <!-- Dynamic Provider Order Preference -->
      <div class="provider-container">
        <label>
          <span>Provider Order Preference</span>
          <span id="lbl-providers-count" class="param-note" style="color:var(--text-subtle);">Fetching supported providers...</span>
        </label>
        
        <div class="provider-add-row">
          <select id="provider-select">
            <option value="">-- Add a Supported Provider --</option>
          </select>
          <button type="button" id="btn-add-provider" onclick="addProvider()">+ Add</button>
        </div>

        <!-- Drag-and-drop provider priority list -->
        <div class="provider-list" id="provider-list-items">
          <!-- Populated dynamically -->
        </div>
      </div>
    </section>

    <!-- 3. Generated URLs -->
    <section>
      <h2>3. Generated Endpoints</h2>

      <div class="endpoint-item">
        <label>Path-Based Base URL (OpenAI SDK / Cursor / Aider / Continue)</label>
        <div class="endpoint-row">
          <div class="code-box" id="path-url">http://localhost:18080/p/anthropic:claude-3.7-sonnet/v1</div>
          <button onclick="copyElement('path-url')">Copy</button>
        </div>
      </div>

      <div class="endpoint-item">
        <label>Query-String Base URL</label>
        <div class="endpoint-row">
          <div class="code-box" id="query-url">http://localhost:18080/v1?model=anthropic/claude-3.7-sonnet&temperature=0.7</div>
          <button onclick="copyElement('query-url')">Copy</button>
        </div>
      </div>
    </section>

    <!-- 4. Test Console -->
    <section>
      <div style="display: flex; justify-content: space-between; align-items: baseline;">
        <h2>4. Live Test Endpoint</h2>
        <span id="test-time" style="font-family: var(--mono); font-size: 13px; color: var(--text-muted);"></span>
      </div>

      <div class="test-box">
        <input type="text" id="test-input" value="Explain quantum superposition in one clear sentence.">
        <button id="test-btn" onclick="sendTest()">Send</button>
      </div>

      <div class="test-output" id="test-output">Ready. Click Send to test proxy streaming.</div>
    </section>

  </div>

  <div id="toast">Copied to clipboard</div>

  <script>
    let allModels = [];
    let currentModelObj = null;
    let selectedModel = 'anthropic/claude-3.7-sonnet';
    let supportedModelProviders = []; // providers serving the currently selected model
    let selectedProviders = []; // user prioritized list

    document.addEventListener('DOMContentLoaded', () => {
      checkStatus();
      loadModels();
      bindInputs();
      initDragAndDrop();
    });

    async function checkStatus() {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        const el = document.getElementById('status-indicator');
        if (data.has_api_key) {
          el.className = 'status online';
          el.textContent = `● Online (${data.masked_key})`;
        } else {
          el.className = 'status warn';
          el.textContent = '● Online (Add OPENROUTER_API_KEY to .env)';
        }
      } catch (e) {
        const el = document.getElementById('status-indicator');
        el.className = 'status';
        el.textContent = '○ Offline';
      }
    }

    async function loadModels() {
      try {
        const res = await fetch('/api/models');
        const json = await res.json();
        allModels = json.data || [];
        renderModelList(allModels);

        const initial = allModels.find(m => m.id === selectedModel) || allModels[0];
        if (initial) pickModel(initial.id);
      } catch (e) {
        document.getElementById('models-list').innerHTML = '<div style="padding:14px;color:var(--text-subtle);">Failed to fetch models from OpenRouter.</div>';
      }
    }

    function renderModelList(list) {
      const box = document.getElementById('models-list');
      if (!list.length) {
        box.innerHTML = '<div style="padding:14px;color:var(--text-subtle);">No matching models found.</div>';
        return;
      }
      box.innerHTML = list.slice(0, 100).map(m => {
        const isSel = m.id === selectedModel;
        const ctx = Math.round((m.context_length || 0) / 1000);
        return `
          <div class="model-option ${isSel ? 'active' : ''}" onclick="pickModel('${m.id}')">
            <span>${m.id}</span>
            <span class="model-ctx">${ctx}k</span>
          </div>
        `;
      }).join('');
    }

    function pickModel(id) {
      selectedModel = id;
      document.getElementById('selected-model-text').textContent = id;
      currentModelObj = allModels.find(x => x.id === id) || { id: id };
      
      const ctx = Math.round((currentModelObj.context_length || 0) / 1000);
      document.getElementById('selected-model-info').textContent = `${ctx}k context`;

      applyModelAdaptations(currentModelObj);
      loadModelProviders(id);
      renderModelList(filterModelsList());
      refreshUrls();
    }

    function applyModelAdaptations(model) {
      if (!model) return;

      // 1. Max Output Tokens: jump directly to the max allowed by the model
      const maxCompletion = model.top_provider?.max_completion_tokens || 
                            model.per_request_limits?.max_tokens || 
                            (model.context_length ? Math.min(model.context_length, 32768) : 8192);
      
      const maxTokInput = document.getElementById('input-maxtok');
      const maxTokLabel = document.getElementById('lbl-maxtok-limit');
      
      maxTokInput.max = maxCompletion;
      maxTokInput.value = maxCompletion; // JUMP TO MAX ALLOWED!
      maxTokLabel.textContent = `(max: ${maxCompletion.toLocaleString()})`;

      // 2. Reasoning Effort: DISABLED with note if not supported (NOT hidden)
      const supportedParams = model.supported_parameters || [];
      const isReasoning = supportedParams.includes('reasoning') || 
                          supportedParams.includes('include_reasoning') ||
                          model.id.includes('r1') || 
                          model.id.includes('o1') || 
                          model.id.includes('o3') || 
                          model.id.includes('claude-3.7-sonnet') ||
                          model.id.includes('thinking') ||
                          model.id.includes('qwq');

      const reasoningSelect = document.getElementById('input-reasoning');
      const reasoningNote = document.getElementById('lbl-reasoning-note');
      
      if (isReasoning) {
        reasoningSelect.disabled = false;
        reasoningNote.className = 'param-note note-reasoning';
        reasoningNote.textContent = 'Supported';
      } else {
        reasoningSelect.disabled = true;
        reasoningSelect.value = 'none';
        reasoningNote.className = 'param-note note-disabled';
        reasoningNote.textContent = '(Not supported by model)';
      }

      // 3. Temperature: DISABLED with note if not supported
      const tempInput = document.getElementById('input-temp');
      const tempNote = document.getElementById('lbl-temp-note');
      if (supportedParams.length > 0 && !supportedParams.includes('temperature')) {
        tempInput.disabled = true;
        tempNote.className = 'param-note note-disabled';
        tempNote.textContent = '(Fixed / not supported)';
      } else {
        tempInput.disabled = false;
        tempNote.className = 'param-note';
        tempNote.innerHTML = `<span id="lbl-temp">${tempInput.value}</span>`;
      }
    }

    async function loadModelProviders(modelId) {
      const countLabel = document.getElementById('lbl-providers-count');
      const select = document.getElementById('provider-select');
      countLabel.textContent = 'Loading providers...';
      select.innerHTML = '<option value="">Loading...</option>';

      try {
        const res = await fetch(`/api/models/${encodeURIComponent(modelId)}/endpoints`);
        const data = await res.json();
        supportedModelProviders = data.providers || [];
        
        if (supportedModelProviders.length === 0 && "/" in modelId) {
          supportedModelProviders = [modelId.split("/")[0].toUpperCase()];
        }

        countLabel.textContent = `${supportedModelProviders.length} providers available for this model`;

        // Filter selectedProviders to only keep ones supported by this new model
        selectedProviders = selectedProviders.filter(p => supportedModelProviders.includes(p));
        renderProviderList();

        // Populate dropdown with this model's actual supported providers
        populateProviderDropdown();
        refreshUrls();
      } catch (e) {
        countLabel.textContent = 'Default routing';
        select.innerHTML = '<option value="">-- No providers listed --</option>';
      }
    }

    function populateProviderDropdown() {
      const select = document.getElementById('provider-select');
      const available = supportedModelProviders.filter(p => !selectedProviders.includes(p));

      if (available.length === 0) {
        select.innerHTML = '<option value="">All supported providers added</option>';
        document.getElementById('btn-add-provider').disabled = true;
      } else {
        select.innerHTML = '<option value="">-- Add a Supported Provider --</option>' + 
          available.map(p => `<option value="${p}">${p}</option>`).join('');
        document.getElementById('btn-add-provider').disabled = false;
      }
    }

    function filterModelsList() {
      const q = document.getElementById('model-search').value.toLowerCase().trim();
      return allModels.filter(m => !q || m.id.toLowerCase().includes(q) || (m.name && m.name.toLowerCase().includes(q)));
    }

    function bindInputs() {
      document.getElementById('model-search').addEventListener('input', () => {
        renderModelList(filterModelsList());
      });

      document.getElementById('input-temp').addEventListener('input', (e) => {
        const lbl = document.getElementById('lbl-temp');
        if (lbl) lbl.textContent = e.target.value;
        refreshUrls();
      });
      document.getElementById('input-topp').addEventListener('input', (e) => {
        document.getElementById('lbl-topp').textContent = e.target.value;
        refreshUrls();
      });
      document.getElementById('input-maxtok').addEventListener('input', refreshUrls);
      document.getElementById('input-reasoning').addEventListener('change', refreshUrls);
    }

    // Provider Management
    function addProvider() {
      const select = document.getElementById('provider-select');
      const val = select.value;
      if (!val) return;
      if (!selectedProviders.includes(val)) {
        selectedProviders.push(val);
        renderProviderList();
        populateProviderDropdown();
        refreshUrls();
      }
    }

    function removeProvider(index) {
      selectedProviders.splice(index, 1);
      renderProviderList();
      populateProviderDropdown();
      refreshUrls();
    }

    function moveProvider(index, direction) {
      const target = index + direction;
      if (target < 0 || target >= selectedProviders.length) return;
      const temp = selectedProviders[index];
      selectedProviders[index] = selectedProviders[target];
      selectedProviders[target] = temp;
      renderProviderList();
      refreshUrls();
    }

    function renderProviderList() {
      const listEl = document.getElementById('provider-list-items');
      if (selectedProviders.length === 0) {
        listEl.innerHTML = '<div style="font-size:13px;color:var(--text-subtle);padding:4px 0;">No priority order set (OpenRouter automatically load-balances).</div>';
        return;
      }

      listEl.innerHTML = selectedProviders.map((p, idx) => `
        <div class="provider-item" draggable="true" data-index="${idx}">
          <div style="display:flex;align-items:center;">
            <span class="provider-drag-handle">⠿</span>
            <span style="font-family:var(--mono);font-size:14px;">${idx + 1}. ${p}</span>
          </div>
          <div class="provider-actions">
            <button type="button" class="btn-small" onclick="moveProvider(${idx}, -1)">▲</button>
            <button type="button" class="btn-small" onclick="moveProvider(${idx}, 1)">▼</button>
            <button type="button" class="btn-small" onclick="removeProvider(${idx})">✕</button>
          </div>
        </div>
      `).join('');

      initDragAndDrop();
    }

    function initDragAndDrop() {
      const items = document.querySelectorAll('.provider-item');
      let draggedItem = null;

      items.forEach(item => {
        item.addEventListener('dragstart', (e) => {
          draggedItem = item;
          setTimeout(() => item.classList.add('dragging'), 0);
        });

        item.addEventListener('dragend', () => {
          if (draggedItem) draggedItem.classList.remove('dragging');
          draggedItem = null;
        });

        item.addEventListener('dragover', (e) => {
          e.preventDefault();
        });

        item.addEventListener('drop', (e) => {
          e.preventDefault();
          if (!draggedItem || draggedItem === item) return;
          const fromIdx = parseInt(draggedItem.dataset.index, 10);
          const toIdx = parseInt(item.dataset.index, 10);
          
          const moved = selectedProviders.splice(fromIdx, 1)[0];
          selectedProviders.splice(toIdx, 0, moved);
          renderProviderList();
          refreshUrls();
        });
      });
    }

    function getValues() {
      const tempInput = document.getElementById('input-temp');
      const reasoningSelect = document.getElementById('input-reasoning');
      
      return {
        model: selectedModel,
        temp: !tempInput.disabled ? parseFloat(tempInput.value) : undefined,
        topP: parseFloat(document.getElementById('input-topp').value),
        maxTok: parseInt(document.getElementById('input-maxtok').value, 10),
        reasoning: !reasoningSelect.disabled ? reasoningSelect.value : 'none',
        providers: selectedProviders.join(',')
      };
    }

    function refreshUrls() {
      const origin = window.location.origin;
      const v = getValues();
      const slug = v.model.replace('/', ':');

      // 1. Path-based URL
      const pathUrl = `${origin}/p/${slug}/v1`;
      document.getElementById('path-url').textContent = pathUrl;

      // 2. Query string URL
      const q = [`model=${encodeURIComponent(v.model)}`];
      if (v.temp !== undefined && v.temp !== 0.7) q.push(`temperature=${v.temp}`);
      if (v.topP !== 1.0) q.push(`top_p=${v.topP}`);
      if (v.maxTok && v.maxTok !== 4096) q.push(`max_tokens=${v.maxTok}`);
      if (v.reasoning && v.reasoning !== 'none') q.push(`reasoning_effort=${v.reasoning}`);
      if (v.providers) q.push(`provider_order=${encodeURIComponent(v.providers)}`);

      document.getElementById('query-url').textContent = `${origin}/v1?${q.join('&')}`;
    }

    async function sendTest() {
      const btn = document.getElementById('test-btn');
      const input = document.getElementById('test-input');
      const output = document.getElementById('test-output');
      const timeEl = document.getElementById('test-time');
      const v = getValues();
      const prompt = input.value.trim();
      if (!prompt) return;

      btn.disabled = true;
      btn.textContent = '...';
      output.textContent = '';
      timeEl.textContent = '';

      const start = performance.now();
      const slug = v.model.replace('/', ':');

      try {
        const payload = {
          messages: [{ role: 'user', content: prompt }],
          top_p: v.topP,
          max_tokens: v.maxTok,
          stream: true
        };
        if (v.temp !== undefined) payload.temperature = v.temp;
        if (v.reasoning !== 'none') payload.reasoning_effort = v.reasoning;
        if (v.providers) {
          payload.extra_body = { provider: { order: selectedProviders } };
        }

        const res = await fetch(`/p/${slug}/v1/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer sk-proxy-local'
          },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          output.textContent = `HTTP ${res.status}: ${await res.text()}`;
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let full = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.replace('data: ', '').trim();
              if (dataStr === '[DONE]') continue;
              try {
                const parsed = JSON.parse(dataStr);
                const delta = parsed.choices?.[0]?.delta?.content || '';
                full += delta;
                output.textContent = full;
              } catch (e) {}
            }
          }
        }
        const took = ((performance.now() - start) / 1000).toFixed(2);
        timeEl.textContent = `${took}s`;
      } catch (e) {
        output.textContent = `Error: ${e.message}`;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Send';
      }
    }

    function copyElement(id) {
      const text = document.getElementById(id).textContent.trim();
      navigator.clipboard.writeText(text);
      const toast = document.getElementById('toast');
      toast.style.opacity = '1';
      setTimeout(() => { toast.style.opacity = '0'; }, 1500);
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
