#!/usr/bin/env bash
set -e

# ==============================================================================
# OpenRouter LiteLLM Proxy Installer & Launchd Daemon Setup
# ==============================================================================

SERVICE_LABEL="com.openrouter.proxy"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOG_DIR="${HOME}/Library/Logs/openrouter-proxy"
PLIST_DEST="${HOME}/Library/LaunchAgents/${SERVICE_LABEL}.plist"
ENV_FILE="${SCRIPT_DIR}/.env"

echo "========================================================"
echo "⚡ Installing OpenRouter LiteLLM Proxy Service (macOS)  "
echo "========================================================"

# 1. Check Python 3
if ! command -v python3 &>/dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH."
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Found Python ${PYTHON_VERSION} ($(which python3))"

# 2. Setup Virtual Environment
if [ ! -d "${VENV_DIR}" ]; then
    echo "📦 Creating virtual environment at .venv..."
    python3 -m venv "${VENV_DIR}"
fi

echo "📦 Installing / updating dependencies in virtual environment..."
"${VENV_DIR}/bin/pip" install --upgrade pip -q
"${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt" -q
echo "✓ Python dependencies installed."

# 3. Setup .env
if [ ! -f "${ENV_FILE}" ]; then
    echo "⚙️ Creating .env configuration from template..."
    cp "${SCRIPT_DIR}/.env.example" "${ENV_FILE}"
    echo "⚠️ NOTE: Please edit ${ENV_FILE} to set your OPENROUTER_API_KEY."
else
    echo "✓ Found existing .env file."
fi

# Load PORT and HOST defaults from .env
PORT="18080"
HOST="0.0.0.0"
if [ -f "${ENV_FILE}" ]; then
    ENV_PORT=$(grep -E "^PORT=" "${ENV_FILE}" | cut -d '=' -f2- | tr -d ' "')
    ENV_HOST=$(grep -E "^HOST=" "${ENV_FILE}" | cut -d '=' -f2- | tr -d ' "')
    [ -n "${ENV_PORT}" ] && PORT="${ENV_PORT}"
    [ -n "${ENV_HOST}" ] && HOST="${ENV_HOST}"
fi

# 4. Create Log Directory
mkdir -p "${LOG_DIR}"
mkdir -p "${HOME}/Library/LaunchAgents"
echo "✓ Log directory configured at: ${LOG_DIR}"

# 5. Generate launchd plist from template
PYTHON_BIN="${VENV_DIR}/bin/python"
SERVER_SCRIPT="${SCRIPT_DIR}/server.py"

sed \
    -e "s|__LABEL__|${SERVICE_LABEL}|g" \
    -e "s|__PYTHON_BIN__|${PYTHON_BIN}|g" \
    -e "s|__SERVER_SCRIPT__|${SERVER_SCRIPT}|g" \
    -e "s|__WORKING_DIR__|${SCRIPT_DIR}|g" \
    -e "s|__LOG_DIR__|${LOG_DIR}|g" \
    -e "s|__PORT__|${PORT}|g" \
    -e "s|__HOST__|${HOST}|g" \
    "${SCRIPT_DIR}/com.openrouter.proxy.plist.template" > "${PLIST_DEST}"

chmod 644 "${PLIST_DEST}"
echo "✓ Generated launchd plist: ${PLIST_DEST}"

# 6. Unload existing service if loaded
if launchctl list | grep -q "${SERVICE_LABEL}"; then
    echo "🔄 Unloading previous service instance..."
    launchctl unload -w "${PLIST_DEST}" 2>/dev/null || true
    sleep 1
fi

# 7. Load and start daemon
echo "🚀 Loading launchd agent..."
launchctl load -w "${PLIST_DEST}"

# 8. Verify Server Readiness & Health
echo "⏳ Waiting for server to become ready on http://localhost:${PORT}..."
MAX_ATTEMPTS=15
ATTEMPT=0
SERVER_READY=false

while [ ${ATTEMPT} -lt ${MAX_ATTEMPTS} ]; do
    if curl -s "http://127.0.0.1:${PORT}/api/health" &>/dev/null; then
        SERVER_READY=true
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    sleep 1
done

if [ "${SERVER_READY}" = true ]; then
    echo ""
    echo "========================================================"
    echo "🎉 OpenRouter Proxy Daemon successfully installed & active!"
    echo "========================================================"
    echo "🌐 Web Dashboard: http://localhost:${PORT}"
    echo "📄 Base Proxy:    http://localhost:${PORT}/p/<model_slug>/v1"
    echo "📋 Logs:          ${LOG_DIR}/stdout.log"
    echo "                  ${LOG_DIR}/stderr.log"
    echo ""
    echo "Manage service anytime using:"
    echo "  ./manage.sh status   - Check service status"
    echo "  ./manage.sh logs     - Stream daemon logs"
    echo "  ./manage.sh restart  - Restart daemon"
    echo "  ./manage.sh stop     - Stop daemon"
    echo "  ./uninstall.sh       - Completely remove daemon"
    echo "========================================================"
    
    # Automatically open configuration dashboard in default browser
    if command -v open &>/dev/null; then
        echo "🚀 Opening configuration dashboard in browser..."
        open "http://localhost:${PORT}"
    fi
else
    echo "⚠️ Warning: Service was loaded, but health endpoint did not respond within ${MAX_ATTEMPTS}s."
    echo "Check logs at: ${LOG_DIR}/stderr.log"
fi
