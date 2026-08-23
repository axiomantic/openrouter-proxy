#!/usr/bin/env bash
set -e

SERVICE_LABEL="com.openrouter.proxy"
PLIST_DEST="${HOME}/Library/LaunchAgents/${SERVICE_LABEL}.plist"
LOG_DIR="${HOME}/Library/Logs/openrouter-proxy"

echo "========================================================"
echo "🛑 Uninstalling OpenRouter LiteLLM Proxy Launchd Daemon"
echo "========================================================"

if [ -f "${PLIST_DEST}" ]; then
    echo "Stopping and unloading launchd service..."
    launchctl unload -w "${PLIST_DEST}" 2>/dev/null || true
    rm -f "${PLIST_DEST}"
    echo "✓ Removed ${PLIST_DEST}"
else
    echo "ℹ️ Plist not found at ${PLIST_DEST}"
fi

read -p "Do you also want to remove log files in ${LOG_DIR}? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "${LOG_DIR}"
    echo "✓ Removed logs directory."
fi

echo "========================================================"
echo "✓ OpenRouter Proxy Daemon uninstalled successfully."
echo "========================================================"
