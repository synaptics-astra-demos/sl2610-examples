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

echo "Installing PortAudio libraries from ${PORTAUDIO_TGZ}..."
tar -xzf "${PORTAUDIO_TGZ}" -C /

echo "Verifying installed libraries..."
ERRORS=0
while IFS= read -r entry; do
    # Skip directories and symlinks — only check real files
    [[ "$entry" == */ ]] && continue
    dest="/${entry}"
    if [[ -L "$dest" ]]; then
        continue
    fi
    if [[ ! -f "$dest" ]]; then
        echo "ERROR: $dest is missing"
        ERRORS=$((ERRORS + 1))
    elif [[ ! -s "$dest" ]]; then
        echo "ERROR: $dest is empty (0 bytes)"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK: $dest ($(wc -c < "$dest") bytes)"
    fi
done < <(tar -tzf "${PORTAUDIO_TGZ}")

if [[ $ERRORS -gt 0 ]]; then
    echo "ERROR: $ERRORS file(s) failed to install correctly."
    df -h /
    exit 1
fi

echo "Done."
