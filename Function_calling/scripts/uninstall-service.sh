#!/bin/bash
# Stop, disable, and remove the FunctionGemma demo systemd service.

set -euo pipefail

SERVICE_NAME="functiongemma-demo.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

log() { printf '\033[1;32m[uninstall-service]\033[0m %s\n' "$*"; }

if [ "$(id -u)" -ne 0 ]; then
    echo "must run as root" >&2
    exit 1
fi

if systemctl list-unit-files "${SERVICE_NAME}" --no-legend | grep -q "${SERVICE_NAME}"; then
    log "stopping ${SERVICE_NAME}"
    systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
    log "disabling ${SERVICE_NAME}"
    systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
fi

if [ -f "${SERVICE_PATH}" ]; then
    log "removing ${SERVICE_PATH}"
    rm -f "${SERVICE_PATH}"
fi

systemctl daemon-reload
log "done."
