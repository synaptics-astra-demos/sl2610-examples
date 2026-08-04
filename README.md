# Synaptics Astra SL2610 Series AI Examples

This repository provides AI example applications for the **Synaptics Astra SL2610** series with **Synaptics Torq with Coral NPU**. Follow the instructions below to set up your environment and run various AI examples in few minutes. Most examples offer a **headless** or **display** version.

## Supported Hardware
- [Astra Machina SL2619 Development Kit](https://www.synaptics.com/products/embedded-processors/sl2610-product-line#devKit)
- [Synaptics Coralboard SL2619](https://developers.google.com/coral/products/SL2610-dev-board)


## Learn more about Synaptics Astra by visiting:

- [Astra](https://www.synaptics.com/products/embedded-processors) – Explore the Astra AI platform.
- [AI Developer Zone](https://developer.synaptics.com/) – Find step-by-step tutorials and resources.

## Torq Compiler & Runtime

The Torq compiler is based on the MLIR framework and IREE runtime. The examples use the Torq compiler to optimize models to run efficiently on the Torq NPU. See Torq documentation for details.  
    
- [Torq Documentation](https://synaptics-torq.github.io/torq-compiler/v/latest)


## Setting up your hardware

- For the Machina Kit - see the [Setting up the hardware](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html) guide.

- For the Coralboard - see the [Getting Started](https://developers.google.com/coral/products/SL2610-dev-board) guide. 


## 🔧 Installation
 
### Connect to the SL2610 

Power up the SL2610 kit and open a terminal - using ADB or other method. See hardware setup guide for details.    

### Connect to the network

To enable online example updates and installation of dependencies, it is recommended to connect the kit to the network. See hardware setup guide for details. 

### Clone the Repository

Clone the repository and its torq-examples submodule using the following command:

```bash
git clone --recurse-submodules https://github.com/synaptics-astra-demos/sl2610-examples
```
Navigate to the Repository Directory:

```bash
cd sl2610-examples
```

If you already cloned the repository without submodules, initialize them once:

```bash
git submodule update --init --recursive
```

### Setup Python Environment

To get started, set up your Python environment. This step ensures all required dependencies are installed and isolated within a virtual environment:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

Install dependencies

```bash
pip install --upgrade pip

pip install https://github.com/synaptics-torq/torq-examples/releases/download/torq-runtime-v2.0-alpha/torq_runtime-2.0.0a1-cp312-cp312-manylinux_2_28_aarch64.whl

pip install -r requirements.txt
```

> [!NOTE]
> This error message can be safely ignored.

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torq-runtime 2.0.0 requires numpy>2.0.0b1, but you have numpy 1.26.4 which is incompatible.
```

## Device Configuration

Some demos require kernel modules, native libraries, or display configuration to be installed on the device. 


Installer scripts are provided:


| Script | Type | Description | Require Reboot? | 
|--------|--------|-------------|------|
| `patch_kernel.sh` | Required | Update the NPU kernel module (`syna_npu.ko`) | Yes | 
| `install_portaudio.sh` | Required | Install PortAudio shared libraries for microphone demos | No | 
| `portrait_setup.sh` | Optional | Configure portrait display orientation (recommended if you connect a MIPI DSI display)| No | 
| `patch_usb_cdc.sh` | Optional | Install USB CDC/serial modules (For Neopixel controller support) | Yes | 

Run the scripts one at a time:

Required:
```bash
./setup/install_portaudio.sh
./setup/patch_kernel.sh
```

Optional:
```bash
./setup/portrait_setup.sh
./setup/patch_usb_cdc.sh
```


## Examples

Check out the README.md files in each of these example directories to run the examples. 

- object_detection - detect objects with YoloV8 - [README.md](object_detection/README.md)
- image_classification - classify images with MobileNetV2 - [README.md](image_classification/README.md)
- speech_to_text - capture speech with Moonshine - [README.md](speech_to_text/README.md)
- gemma_translation - translate text to other languages using Gemma3 - [README.md](gemma_translate/README.md)
- function_calling - control the system with voice/text - [README.md](Function_calling/README.md)


## Auto-start with Systemd (GUI Demos)

For the GUI versions of the demos (Object Detection, Gemma Translate, etc), systemd unit templates are provided in each demo's directory. These units assume the project is installed at `/home/root/sl2610-examples` by default.

To generate and install a service:
```bash
cd <demo_dir>
bash scripts/install-service.sh [--root /path/to/sl2610-examples]
```

This will generate a `.service` file from the template and, if run as root on the target device, install it to `/etc/systemd/system/`.


## Resources

- [Astra](https://www.synaptics.com/products/embedded-processors) – Explore the Astra AI platform.
- [AI Developer Zone](https://developer.synaptics.com/) – Find step-by-step tutorials and resources.
- [Astra SDK (Linux)](https://synaptics-astra.github.io/doc/v/latest/) - Customize your image. 
- [Torq Compiler Documentation](https://synaptics-torq.github.io/torq-compiler/v/latest) - Compile and optimize models for powerful and efficient Edge AI performance. 

## Build Something Awesome!
Great technology becomes meaningful when used to make the world a better place. We are here to support you.

If you are stuck or find an issue, please raise a support ticket in the [Astra Support Portal](https://synacsm.atlassian.net/servicedesk/customer/portal/543).
