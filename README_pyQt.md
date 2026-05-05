# PyQt UI variant

A simplified PyQt5 UI variant of the demo, intended for the 7" Wayland
panel on the Coral Dev Board. Top half shows live system metrics with
sparklines; bottom half shows a scrolling log of natural-language prompts
+ the tool calls they produced.

- Follow the steps in [`README.md`](./README.md) (clone, venv, deps, model download)
- Source the Wayland environment so Qt finds the panel:

```bash
source ./setup_wayland.sh
```

- Run the UI from `src/`:

```bash
cd src
python3 app_pyqt.py
```

- With the optional Mini Sparkle Motion ring attached:

```bash
python3 app_pyqt.py --wled-port /dev/ttyACM0
```

- For full-screen on the 7" panel:

```bash
python3 app_pyqt.py --fullscreen
```

Press `Ctrl+P` for a screenshot to `/tmp/`. Press `Esc` to quit.
