#!/bin/bash
set -e

# Instructions
#   chmod +x portrait_setup.sh
#   ./portrait_setup.sh

DEST="/etc/xdg/weston/weston.ini"
BACKUP="${DEST}.bak"

[[ -f "$DEST" ]] || { echo "ERROR: $DEST not found"; exit 1; }

# Back up once so the original can be restored if needed
if [[ ! -f "$BACKUP" ]]; then
    cp "$DEST" "$BACKUP"
    echo "Backed up original weston.ini to $BACKUP"
fi

# [core]: disable Synaptics desktop shell for portrait/demo mode
sed -i 's|^shell=|#shell=|' "$DEST"

# [shell]: hide desktop UI elements
sed -i 's|^client=|#client=|'                              "$DEST"
sed -i 's|^background-image=|#background-image=|'          "$DEST"
sed -i 's|^background-type=|#background-type=|'            "$DEST"
sed -i 's|^clock-format=|#clock-format=|'                  "$DEST"
sed -i 's|^panel-color=|#panel-color=|'                    "$DEST"
sed -i 's|^panel-position=bottom|panel-position=none|'     "$DEST"
sed -i 's|^img-panel=|#img-panel=|'                        "$DEST"
sed -i 's|^custom-image=|#custom-image=|'                  "$DEST"
sed -i 's|^binding-modifier=|#binding-modifier=|'          "$DEST"

# [output]: add portrait rotation (idempotent — skipped if already set)
grep -q '^transform=' "$DEST" || sed -i '/^mode=800x480/a transform=rotate-270' "$DEST"

echo "Patched $DEST with portrait settings"

# Environment variables for apps
sed -i '/^export ORIENTATION=/d'      /etc/profile
sed -i '/^export DISPLAY_WIDTH=/d'    /etc/profile
sed -i '/^export DISPLAY_HEIGHT=/d'   /etc/profile
sed -i '/^export XDG_RUNTIME_DIR=/d'  /etc/profile
sed -i '/^export WAYLAND_DISPLAY=/d'  /etc/profile
echo 'export ORIENTATION=portrait'             >> /etc/profile
echo 'export DISPLAY_WIDTH=480'                >> /etc/profile
echo 'export DISPLAY_HEIGHT=800'               >> /etc/profile
echo 'export XDG_RUNTIME_DIR=/var/run/user/0'  >> /etc/profile
echo 'export WAYLAND_DISPLAY=wayland-1'        >> /etc/profile
echo "Updated /etc/profile"

systemctl restart weston
sleep 3
if ! systemctl is-active --quiet weston; then
    echo "ERROR: Weston failed to start. Restoring original weston.ini ..."
    cp "$BACKUP" "$DEST"
    systemctl restart weston
    echo "Restored. Check: journalctl -u weston -n 50"
    exit 1
fi

echo "Portrait display configured successfully."
