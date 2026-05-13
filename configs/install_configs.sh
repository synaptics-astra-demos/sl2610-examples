#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.
#
# Install device configurations for sl2610-examples.
#
# Usage:
#   sudo ./install_configs.sh <target> [target ...]
#
# Targets:
#   kernel          - Install NPU kernel module (patch_kernel.sh)
#   usb_cdc         - Install USB CDC/serial modules (patch_usb_cdc.sh)
#   portaudio       - Install PortAudio shared libraries for microphone demos
#   portrait_setup  - Configure portrait display orientation
#   all             - Install all of the above
#
# Examples:
#   sudo ./install_configs.sh kernel
#   sudo ./install_configs.sh usb_cdc portaudio portrait_setup
#   sudo ./install_configs.sh all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ALL_TARGETS=(kernel usb_cdc portaudio portrait_setup)

# --- helpers ----------------------------------------------------------------

usage() {
    echo "Usage: sudo $0 <target> [target ...]"
    echo ""
    echo "Targets:"
    echo "  kernel          Install NPU kernel module"
    echo "  usb_cdc         Install USB CDC/serial kernel modules"
    echo "  portaudio       Install PortAudio shared libraries for microphone demos"
    echo "  portrait_setup  Configure portrait display orientation"
    echo "  all             Install all targets"
    exit 1
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "ERROR: This script must be run as root (sudo)."
        exit 1
    fi
}

# --- target functions -------------------------------------------------------

install_kernel() {
    echo "=== Installing NPU kernel module ==="
    SKIP_REBOOT=1 bash "$SCRIPT_DIR/patch_kernel.sh" "$SCRIPT_DIR"
}

install_usb_cdc() {
    echo "=== Installing USB CDC/serial modules ==="
    SKIP_REBOOT=1 bash "$SCRIPT_DIR/patch_usb_cdc.sh" "$SCRIPT_DIR"
}

install_portaudio() {
    echo "=== Installing PortAudio shared libraries ==="
    bash "$SCRIPT_DIR/install_portaudio.sh"
}

install_portrait_setup() {
    echo "=== Configuring portrait display ==="
    bash "$SCRIPT_DIR/portrait_setup.sh"
}

# --- main -------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    usage
fi

check_root

TARGETS=()
for arg in "$@"; do
    case "$arg" in
        all)
            TARGETS=("${ALL_TARGETS[@]}")
            break
            ;;
        kernel|usb_cdc|portaudio|portrait_setup)
            TARGETS+=("$arg")
            ;;
        *)
            echo "ERROR: Unknown target '$arg'"
            usage
            ;;
    esac
done

# Deduplicate while preserving order
declare -A SEEN
UNIQUE_TARGETS=()
for t in "${TARGETS[@]}"; do
    if [[ -z "${SEEN[$t]:-}" ]]; then
        SEEN[$t]=1
        UNIQUE_TARGETS+=("$t")
    fi
done

REBOOT_NEEDED=false

for target in "${UNIQUE_TARGETS[@]}"; do
    case "$target" in
        kernel)
            install_kernel
            REBOOT_NEEDED=true
            ;;
        usb_cdc)
            install_usb_cdc
            REBOOT_NEEDED=true
            ;;
        portaudio)
            install_portaudio
            ;;
        portrait_setup)
            install_portrait_setup
            ;;
    esac
done

echo ""
echo "=== All requested targets installed ==="
if [[ "$REBOOT_NEEDED" == true ]]; then
    echo "A reboot is required. Reboot now? [y/N]"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        reboot
    else
        echo "Skipping reboot. Remember to reboot manually."
    fi
fi
