#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-$(dirname "$(realpath "$0")")}"
KMOD="/lib/modules/$(uname -r)/kernel"

for f in cdc-acm.ko usbserial.ko ch341.ko; do
    if [[ ! -f "$SRC_DIR/$f" ]]; then
        echo "ERROR: $SRC_DIR/$f not found"
        exit 1
    fi
done

echo "[1/4] Creating module directories..."
mkdir -p "$KMOD/drivers/usb/class" "$KMOD/drivers/usb/serial"

echo "[2/4] Installing kernel modules..."
cp "$SRC_DIR/cdc-acm.ko"   "$KMOD/drivers/usb/class/"
cp "$SRC_DIR/usbserial.ko" "$KMOD/drivers/usb/serial/"
cp "$SRC_DIR/ch341.ko"     "$KMOD/drivers/usb/serial/"

echo "[3/4] Updating module dependencies..."
depmod -a

echo "[4/4] Loading cdc_acm module..."
modprobe cdc_acm

echo "Done."
if [[ "${SKIP_REBOOT:-0}" != "1" ]]; then
    echo "Rebooting..."
    reboot
fi