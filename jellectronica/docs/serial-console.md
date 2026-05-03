# Serial Console Setup

Connect to the Coral Dev Board's debug console via USB-TTL serial adapter.

## What You Need

- USB-TTL serial adapter (e.g., DSD TECH SH-U09C with FTDI FT232RL)
- Baud rate: **115200**, 8N1

> ⚠️ **3.3V logic only!** Set your adapter to 3.3V. Do NOT connect 5V to the board.

## Wiring

Connect the adapter to the **40-pin GPIO header (J32)**:

| Adapter Wire | → | Board Pin | Function |
|---|---|---|---|
| GND | → | Pin 6 | Ground |
| TX | → | Pin 10 | Board RX |
| RX | → | Pin 8 | Board TX |

> Do NOT connect the adapter's VCC/power wire. The board has its own power supply.

## Connect

### macOS

```bash
# Find the serial device
ls /dev/cu.usbserial*

# Connect
screen /dev/cu.usbserial-XXXXXXXX 115200

# Exit: Ctrl+A, K, Y
```

### Linux

```bash
screen /dev/ttyUSB0 115200
```

## Login

```
Login: root
Password: (blank — press Enter)
```

## Verify System

```bash
uname -a                  # Kernel version
python3 --version         # Should be 3.12+
ls /dev/torq              # NPU device
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No output | Check TX/RX are crossover-wired (not straight) |
| Garbage characters | Verify baud rate is 115200 |
| Device not detected | Try `ls /dev/cu.*` or `ls /dev/ttyUSB*` |
| Green LED flashing | Insufficient power — use the supplied 15V USB-PD adapter |
