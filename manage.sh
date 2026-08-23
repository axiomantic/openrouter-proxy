#!/usr/bin/env bash

SERVICE_LABEL="com.openrouter.proxy"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_DEST="${HOME}/Library/LaunchAgents/${SERVICE_LABEL}.plist"
LOG_DIR="${HOME}/Library/Logs/openrouter-proxy"
ENV_FILE="${SCRIPT_DIR}/.env"

PORT="18080"
if [ -f "${ENV_FILE}" ]; then
    ENV_PORT=$(grep -E "^PORT=" "${ENV_FILE}" | cut -d '=' -f2- | tr -d ' "')
    [ -n "${ENV_PORT}" ] && PORT="${ENV_PORT}"
fi

case "$1" in
    status)
        echo "=== Launchd Service Status: ${SERVICE_LABEL} ==="
        if launchctl list | grep "${SERVICE_LABEL}"; then
            echo "Status: Running ✅"
            echo "URL:    http://localhost:${PORT}"
        else
            echo "Status: Stopped / Not Loaded ❌"
        fi
        ;;
    start)
        echo "Starting ${SERVICE_LABEL}..."
        if [ ! -f "${PLIST_DEST}" ]; then
            echo "Error: Plist not found at ${PLIST_DEST}. Run ./install.sh first."
            exit 1
        fi
        launchctl load -w "${PLIST_DEST}"
        echo "Service loaded."
        ;;
    stop)
        echo "Stopping ${SERVICE_LABEL}..."
        if [ -f "${PLIST_DEST}" ]; then
            launchctl unload -w "${PLIST_DEST}" 2>/dev/null || true
            echo "Service unloaded."
        else
            echo "Plist not found."
        fi
        ;;
    restart)
        echo "Restarting ${SERVICE_LABEL}..."
        if [ -f "${PLIST_DEST}" ]; then
            launchctl unload -w "${PLIST_DEST}" 2>/dev/null || true
            sleep 1
            launchctl load -w "${PLIST_DEST}"
            echo "Service restarted."
        else
            echo "Error: Plist not found. Run ./install.sh first."
            exit 1
        fi
        ;;
    logs)
        echo "Streaming logs from ${LOG_DIR} (Ctrl+C to exit)..."
        tail -f "${LOG_DIR}/stdout.log" "${LOG_DIR}/stderr.log"
        ;;
    run)
        echo "Starting server in foreground (debug mode)..."
        if [ -d "${SCRIPT_DIR}/.venv" ]; then
            "${SCRIPT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/server.py"
        else
            python3 "${SCRIPT_DIR}/server.py"
        fi
        ;;
    *)
        echo "Usage: $0 {status|start|stop|restart|logs|run}"
        exit 1
        ;;
esac
