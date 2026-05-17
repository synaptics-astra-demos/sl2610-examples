#!/usr/bin/env bash
set -euo pipefail

SERVICES=(
    swupdate.socket
    swupdate.service
    ModemManager
    bluealsa
)

echo "Stopping unused services (will resume on reboot)..."
for svc in "${SERVICES[@]}"; do
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        systemctl stop "$svc"
        echo "  stopped: $svc"
    else
        echo "  already stopped: $svc"
    fi
done

echo "Done. Free memory:"
free -h
