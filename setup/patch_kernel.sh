#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-$(dirname "$(realpath "$0")")}"
MODULE_PATH="/usr/lib/modules/6.12.62/updates/syna_npu.ko"
EXPECTED_MD5="5da15ae4fa99e6e5af986243fee10b1e"

if [[ ! -f "$SRC_DIR/syna_npu.ko" ]]; then
    echo "ERROR: $SRC_DIR/syna_npu.ko not found"
    exit 1
fi

echo "[1/2] Installing to $MODULE_PATH..."
mkdir -p "$(dirname "$MODULE_PATH")"
cp "$SRC_DIR/syna_npu.ko" "$MODULE_PATH"

echo "[2/2] Verifying md5sum..."
ACTUAL_MD5=$(md5sum "$MODULE_PATH" | awk '{print $1}')
if [[ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]]; then
    echo "ERROR: md5sum mismatch"
    echo "  expected: $EXPECTED_MD5"
    echo "  actual:   $ACTUAL_MD5"
    exit 1
fi
echo "  md5sum OK: $ACTUAL_MD5"

echo "Done."
if [[ "${SKIP_REBOOT:-0}" != "1" ]]; then
    echo "Rebooting..."
    reboot
fi
