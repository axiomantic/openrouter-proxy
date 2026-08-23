# openrouter-proxy ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![macOS launchd](https://img.shields.io/badge/platform-macOS%20launchd-lightgrey.svg)](https://www.apple.com/macos/)
[![LiteLLM](https://img.shields.io/badge/powered%20by-LiteLLM-purple.svg)](https://github.com/BerriAI/litellm)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com)

A lightweight, high-performance OpenAI-compatible proxy daemon for macOS that routes through [OpenRouter](https://openrouter.ai/) using [LiteLLM](https://github.com/BerriAI/litellm).

It allows dynamically overriding any model configuration (model slug, temperature, max tokens, top-p, reasoning effort, custom system prompts, provider fallbacks) directly via **URL query parameters** or **path components**, accompanied by a built-in configuration dashboard with real-time model discovery and live prompt testing.

---

## 🎯 Key Features

- 🔄 **macOS `launchd` Service**: Runs unobtrusively in the background, starts on login, automatically restarts if stopped.
- 🔀 **Dynamic URL-Based Configuration**:
  - **Path-Based Base URLs**: `http://localhost:18080/p/anthropic:claude-3.7-sonnet/v1`
  - **Query-String Overrides**: `http://localhost:18080/v1?model=deepseek/deepseek-r1&temperature=0.6&max_tokens=4096&reasoning_effort=high`
  - **Base64 Config Payloads**: `http://localhost:18080/cfg/<base64_json>/v1`
- 🖥️ **Interactive Web UI Dashboard** (`http://localhost:18080`):
  - Live model browser dynamically queried from the OpenRouter model registry.
  - Search and filter by provider, reasoning (CoT), multimodal/vision, free tiers, and context length.
  - Interactive sliders for temperature, top-p, max tokens, reasoning effort (low/medium/high), system prompt injection, and provider ordering.
  - Live code snippet & URL generator (cURL, Python `openai`, Node.js, Aider, Cursor, Continue).
  - Built-in interactive streaming test console with TTFT (time-to-first-token) benchmarking.
- 🌐 **Full OpenAI API Compatibility**:
  - `POST /v1/chat/completions` (and `/p/{model}/v1/chat/completions`)
  - `GET /v1/models`
  - Server-Sent Events (SSE) token streaming (`stream: true`).
- 🔐 **Flexible Multi-Tier Auth**:
  - Auto-discovers API key from `.env` in project root, `~/.config/openrouter-proxy/.env`, `~/.env`, or shell `OPENROUTER_API_KEY`.
  - Transparently forwards client-supplied bearer tokens (`sk-or-...`).

---

## 🚀 Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/axiomantic/openrouter-proxy.git
cd openrouter-proxy

# Copy example environment configuration
cp .env.example .env
```

Edit `.env` to set your OpenRouter API key:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
PORT=18080
HOST=0.0.0.0
```

### 2. Install & Start Background Daemon

Run the installer:
```bash
./install.sh
```

This will:
1. Create a dedicated Python virtual environment (`.venv`).
2. Install all required dependencies (`litellm`, `fastapi`, `uvicorn`, `httpx`, `python-dotenv`).
3. Generate and register the macOS `launchd` plist file (`~/Library/LaunchAgents/com.openrouter.proxy.plist`).
4. Start the service automatically.

### 3. Open the Dashboard

Navigate to:
👉 **[http://localhost:18080](http://localhost:18080)**

---

## 🛠️ Service Management

Use `./manage.sh` to control the daemon:

```bash
./manage.sh status   # Check daemon running status & URL
./manage.sh restart  # Restart background daemon
./manage.sh logs     # Follow live stdout and stderr logs
./manage.sh stop     # Stop and unload daemon
./manage.sh start    # Start daemon
./manage.sh run      # Run in foreground for debugging
./uninstall.sh       # Completely remove launchd plist and agent
```

Logs are maintained at:
- `~/Library/Logs/openrouter-proxy/stdout.log`
- `~/Library/Logs/openrouter-proxy/stderr.log`

---

## 🔌 Dynamic URL Configuration Formats

### 1. Path-Based URL (Recommended)
Use colons (`:`) in place of slashes (`/`) for model names to keep URLs clean:

```text
http://localhost:18080/p/anthropic:claude-3.7-sonnet/v1
http://localhost:18080/p/deepseek:deepseek-r1/v1
http://localhost:18080/p/meta-llama:llama-3.3-70b-instruct/v1
http://localhost:18080/p/google:gemini-2.5-pro/v1
```

### 2. Query Parameters
Append parameters to any endpoint to override defaults:

```text
http://localhost:18080/v1/chat/completions?model=anthropic/claude-3.7-sonnet&temperature=0.5&max_tokens=4096&reasoning_effort=high
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `model` | string | Model ID (e.g. `anthropic/claude-3.7-sonnet`) |
| `temperature` / `temp` | float | Sampling temperature (`0.0` - `2.0`) |
| `top_p` | float | Nucleus sampling probability (`0.0` - `1.0`) |
| `max_tokens` | integer | Maximum output token limit |
| `reasoning_effort` | string | Reasoning budget: `low`, `medium`, `high` |
| `system_prompt` | string | Prepend or override system prompt |
| `provider_order` | string | Comma-separated provider preference (e.g. `Together,Fireworks`) |
| `allow_fallbacks` | boolean | Enable/disable provider fallbacks (`true`/`false`) |

---

## 💻 Integration Examples

### Python (`openai` SDK)

```python
from openai import OpenAI

# Target any model dynamically through base_url
client = OpenAI(
    base_url="http://localhost:18080/p/anthropic:claude-3.7-sonnet/v1",
    api_key="sk-proxy-local"  # Server uses OPENROUTER_API_KEY from .env
)

response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### cURL

```bash
curl http://localhost:18080/p/deepseek:deepseek-r1/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proxy-local" \
  -d '{
    "messages": [{"role": "user", "content": "Write a quick haiku about daemons."}],
    "temperature": 0.6,
    "stream": true
  }'
```

### Cursor / Continue.dev / Aider

Set your base URL in environment variables or client settings:
```bash
export OPENAI_BASE_URL="http://localhost:8080/p/anthropic:claude-3.7-sonnet/v1"
export OPENAI_API_KEY="sk-proxy-local"
```

---

## 📄 License

[MIT License](LICENSE) © 2026 Axiomantic
