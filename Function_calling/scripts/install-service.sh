#!/bin/bash
# Install a systemd unit that auto-starts the FunctionGemma PyQt demo on boot.
#
# The unit:
#   - waits for weston (Wayland compositor) to be up
#   - runs <repo>/.venv/bin/python3 app_pyqt.py --fullscreen [extras]
#   - sets the Wayland env vars used by the OOBE image's weston
#   - on stop / crash, drives BUZZERn HIGH so the buzzer can never
#     latch ON across a service exit
#
# Usage:
#   sudo bash scripts/install-service.sh                       # default flags
#   sudo bash scripts/install-service.sh --wled-port /dev/ttyACM0
#   sudo bash scripts/install-service.sh --voice stub
#   sudo bash scripts/install-service.sh --no-enable           # install but don't enable
#   sudo bash scripts/install-service.sh --no-start            # enable but don't start now
#
# Re-running overwrites the existing unit. To remove the service, use
# scripts/uninstall-service.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FC_DIR}/.." && pwd)"
VENV_PY="${REPO_ROOT}/.venv/bin/python3"
SERVICE_NAME="functiongemma-demo.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

ENABLE=1
START=1
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-enable) ENABLE=0; shift ;;
        --no-start)  START=0; shift ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

log() { printf '\033[1;32m[install-service]\033[0m %s\n' "$*"; }

if [ "$(id -u)" -ne 0 ]; then
    echo "must run as root (or under sudo) to write ${SERVICE_PATH}" >&2
    exit 1
fi

if [ ! -x "${VENV_PY}" ]; then
    echo "venv python not found at ${VENV_PY}" >&2
    echo "run scripts/setup.sh first to create the venv and install deps" >&2
    exit 1
fi

if [ ! -f "${FC_DIR}/app_pyqt.py" ]; then
    echo "app_pyqt.py not found at ${FC_DIR}/app_pyqt.py" >&2
    exit 1
fi

EXEC_ARGS="--fullscreen"
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    # quote each extra arg for systemd (semi-naive: assumes no embedded quotes)
    for a in "${EXTRA_ARGS[@]}"; do
        EXEC_ARGS="${EXEC_ARGS} ${a}"
    done
fi

log "writing ${SERVICE_PATH}"
cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=FunctionGemma physical-AI PyQt demo
Documentation=https://github.com/synaptics-astra-demos/sl2610-examples
After=weston.service systemd-user-sessions.service
Wants=weston.service

[Service]
Type=simple
User=root
WorkingDirectory=${FC_DIR}
Environment=XDG_RUNTIME_DIR=/var/run/user/0
Environment=WAYLAND_DISPLAY=wayland-1
Environment=QT_QPA_PLATFORM=wayland
Environment=WESTON_DISABLE_GBM_MODIFIERS=true
Environment=PYTHONUNBUFFERED=1
ExecStart=${VENV_PY} ${FC_DIR}/app_pyqt.py ${EXEC_ARGS}
# Hard-stop safety net: even if the Python process is SIGKILL'd, drive
# BUZZERn back to silent (1 = silent on this HAT — see hardware.py).
ExecStopPost=-/bin/sh -c 'gpioset \$(gpiofind BUZZERn)=1 || true'
TimeoutStartSec=120
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

log "reloading systemd"
systemctl daemon-reload

if [ "${ENABLE}" -eq 1 ]; then
    log "enabling ${SERVICE_NAME}"
    systemctl enable "${SERVICE_NAME}"
fi

if [ "${START}" -eq 1 ] && [ "${ENABLE}" -eq 1 ]; then
    log "starting ${SERVICE_NAME}"
    systemctl restart "${SERVICE_NAME}"
    sleep 2
    systemctl --no-pager --full status "${SERVICE_NAME}" || true
fi

log "done."
log "  status:  systemctl status ${SERVICE_NAME}"
log "  logs:    journalctl -u ${SERVICE_NAME} -f"
log "  remove:  bash ${SCRIPT_DIR}/uninstall-service.sh"
