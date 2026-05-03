# Board Recovery Guide

How to recover the Coral Dev Board when it won't boot.

## Symptoms

The board powers on (green LED) but never reaches Linux. On USB, it may appear as `TinyUSB Device` (VID `0x06CB`, PID `0x019E`).

## What You Need

| Item | Details |
|------|---------|
| Serial adapter | USB-TTL at 115200 baud (see [serial-console.md](serial-console.md)) |
| USB-C data cable | Connected to the board's **OTG port** (not the power port) |
| Factory boot image | `SYNAIMG/boot.subimg.gz` (decompress first) |

## Recovery Steps

### 1. Connect Serial Console

Wire your USB-TTL adapter to the 40-pin header (TX→Pin 10, RX→Pin 8, GND→Pin 6).

### 2. Catch U-Boot Prompt

Start a script that spams keypresses, then power cycle the board:

```bash
python3 -c "
import serial, time
ser = serial.Serial('/dev/cu.usbserial-XXXXXXXX', 115200, timeout=1)
ser.reset_input_buffer()
print('Spamming Enter — POWER CYCLE THE BOARD NOW')
start = time.time()
while time.time() - start < 60:
    ser.write(b' \r\n')
    time.sleep(0.3)
    data = ser.read(4096)
    if data:
        text = data.decode('utf-8', errors='replace')
        print(text, end='', flush=True)
        if b'=>' in data:
            print('\n*** U-BOOT CAUGHT ***')
            break
ser.close()
"
```

**While the script is running**, unplug the power cable, wait 5 seconds, plug it back in.

### 3. Start Fastboot

At the `=>` prompt:

```
fastboot usb 0
```

### 4. Flash Factory Boot Image

On your host machine:

```bash
# Verify fastboot sees the device
fastboot devices

# Decompress factory image
gzip -dk SYNAIMG/boot.subimg.gz -c > /tmp/boot_factory.img

# Flash both slots
fastboot flash boot_a /tmp/boot_factory.img
fastboot flash boot_b /tmp/boot_factory.img
```

### 5. Power Cycle

> **Important**: `fastboot reboot` does NOT reliably reset this board. Physically unplug and replug the power cable.

The board should boot within ~12 seconds. Verify:

```bash
adb devices
```

## A/B Slot Management

The board uses A/B boot slots. To switch slots from U-Boot:

```
bootslot set b     # Switch to slot B
bootslot reset     # Reset boot counter
boot               # Boot immediately
```

From Linux:

```bash
bootctrl set-active-boot-slot 0   # Slot A
bootctrl set-active-boot-slot 1   # Slot B
reboot
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No serial output | Check TX/RX wiring is crossed (adapter TX → board Pin 10) |
| U-Boot boots too fast | Start the spam script **before** plugging in power |
| `fastboot devices` empty | USB-C must be in the **OTG port**, and `fastboot usb 0` must be running |
| Board stuck in bootloader | Press the RESET button or power cycle |
