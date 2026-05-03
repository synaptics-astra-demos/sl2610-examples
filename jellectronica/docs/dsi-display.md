# Waveshare 5" DSI Display Setup

Guide for configuring the Waveshare 5" DSI LCD on the Coral Dev Board.

## Hardware

| Component | Details |
|-----------|---------|
| **Display** | Waveshare 5" DSI LCD (Capacitive Touch) |
| **Resolution** | 800×480 @ 60Hz |
| **Interface** | MIPI DSI (1 data lane) |
| **Touch** | FocalTech FT5506, I2C bus 3, address 0x38 |
| **Backlight** | ATtiny regulator, I2C bus 3, address 0x45 |

## Connection

Connect the 15-pin DSI ribbon cable from the display to the MIPI DSI connector on the Astra I/O board. Touch and backlight use I2C via the same cable.

## Required Kernel Modules

| Module | Purpose |
|--------|---------|
| `panel_waveshare_dsi` | DSI panel driver |
| `edt_ft5x06` | Capacitive touch driver |
| `rpi_panel_attiny_regulator` | Backlight/power regulator |
| `syna_drm` | Synaptics DRM/KMS display driver |

## Weston Configuration

### `/etc/default/weston`
```bash
WESTON_DISABLE_GBM_MODIFIERS=true
BACKEND=drm-backend.so
HOME=/home/root
SEATID=seat0
LD_PRELOAD=/usr/lib/libmali.so
SEATD_VTBOUND=0
```

### `/etc/xdg/weston/weston.ini`
```ini
[core]
require-input=false
idle-time=0
require-outputs=none

[shell]
binding-modifier=alt

[output]
name=DPI-1
mode=800x480

[terminal]
term=vt100
```

### `/etc/udev/rules.d/99-drm.rules`
```
SUBSYSTEM=="drm", MODE="0666"
```

## Critical Fixes

### 1. libmali.so Symlink
The `libmali.so.0` symlink must point to the full Mali blob (with EGL/GBM), not the OpenCL-only build:

```bash
rm -f /usr/lib/libmali.so.0
ln -s /usr/lib/libmali.so.0.49.1 /usr/lib/libmali.so.0
```

### 2. LD_PRELOAD
The DRM backend requires `gbm_bo_get_plane_count` from the Mali blob. Without `LD_PRELOAD=/usr/lib/libmali.so`, Weston fails to start.

### 3. SEATD_VTBOUND=0
The board has no VT subsystem. Without this setting, seatd fails to initialize.

## Backlight Control

```bash
# Read brightness (0-255)
cat /sys/class/backlight/3-0045/brightness

# Set brightness
echo 128 > /sys/class/backlight/3-0045/brightness

# Turn off/on
echo 4 > /sys/class/backlight/3-0045/bl_power   # off
echo 0 > /sys/class/backlight/3-0045/bl_power   # on
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Weston won't start | Check `journalctl -u weston.service -n 50` |
| Display stays black | Verify DPI-1 is "connected": `modetest -M synaptics -c` |
| Touch not responding | Check I2C: `i2cdetect -y 3` (should show 0x38 and 0x45) |

> **Note**: The kernel log may show `mipi_dsih_open:Core Version not supported!!!` — this is a non-fatal warning. The display works correctly via the DPI bridge path.
