#!/bin/bash
# Install a systemd unit that auto-starts the YOLOv8 Object Detection GUI on boot.
#
# Usage:
#   bash scripts/install-service.sh [--root /path/to/sl2610-examples]
#
# Defaults to /home/root/sl2610-examples if run without --root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="object-detection.service"
TEMPLATE_PATH="${DEMO_DIR}/${SERVICE_NAME}.in"
SERVICE_PATH="${DEMO_DIR}/${SERVICE_NAME}"

ROOT_PATH="/home/root/sl2610-examples"

while [ $# -gt 0 ]; do
    case "$1" in
        --root) ROOT_PATH="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "Generating ${SERVICE_NAME} with ROOT=${ROOT_PATH}"

sed "s|@ROOT@|${ROOT_PATH}|g" "${TEMPLATE_PATH}" > "${SERVICE_PATH}"

# If running as root on a systemd system, offer to install it
if [ "$(id -u)" -eq 0 ] && [ -d /etc/systemd/system ]; then
    echo "Installing ${SERVICE_NAME} to /etc/systemd/system/"
    cp "${SERVICE_PATH}" "/etc/systemd/system/"
    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}"
    echo "Done. Use 'systemctl start ${SERVICE_NAME}' to run."
else
    echo "Generated ${SERVICE_PATH}."
    echo "To install manually, copy this file to /etc/systemd/system/ on your target."
fi
