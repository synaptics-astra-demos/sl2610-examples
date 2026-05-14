#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORTAUDIO_TGZ="${1:-${SCRIPT_DIR}/portaudio_libs.tgz}"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root."
    exit 1
fi

if [[ ! -f "${PORTAUDIO_TGZ}" ]]; then
    echo "ERROR: ${PORTAUDIO_TGZ} not found; mic stream will fail unless libportaudio.so.2 is installed system-wide"
    exit 1
fi

echo "[1/2] Installing PortAudio libraries from ${PORTAUDIO_TGZ}..."
tar -xzf "${PORTAUDIO_TGZ}" -C /

echo "[2/2] Updating dynamic linker cache..."
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig
else
    echo "ldconfig not found; skipping cache update."
fi

echo "Done."
