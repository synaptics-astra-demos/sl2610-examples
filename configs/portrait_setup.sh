#!/bin/bash
set -e

# Instructions
#   chmod +x portrait_setup.sh
#   ./portrait_setup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="/etc/xdg/weston/weston.ini"

cp "$SCRIPT_DIR/weston.ini" "$DEST"
echo "Installed weston.ini to $DEST"

sed -i '/^export ORIENTATION=/d' /etc/profile
sed -i '/^export DISPLAY_WIDTH=/d' /etc/profile
sed -i '/^export DISPLAY_HEIGHT=/d' /etc/profile
sed -i '/^export XDG_RUNTIME_DIR=/d' /etc/profile
sed -i '/^export WAYLAND_DISPLAY=/d' /etc/profile
echo 'export ORIENTATION=portrait' >> /etc/profile
echo 'export DISPLAY_WIDTH=480' >> /etc/profile
echo 'export DISPLAY_HEIGHT=800' >> /etc/profile
echo 'export XDG_RUNTIME_DIR=/var/run/user/0' >> /etc/profile
echo 'export WAYLAND_DISPLAY=wayland-1' >> /etc/profile
echo "Set ORIENTATION, DISPLAY_WIDTH, DISPLAY_HEIGHT, XDG_RUNTIME_DIR, WAYLAND_DISPLAY in /etc/profile"
source /etc/profile

systemctl restart weston
echo "Restarted weston"
