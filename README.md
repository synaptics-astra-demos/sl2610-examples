# Synaptics Astra SL2610 Series AI Examples

This repository provides AI example applications for the **Synaptics Astra SL2610** series with **Synaptics Torq with Coral NPU**. Follow the instructions below to set up your environment and run various AI examples in few minutes. Most examples offer a **headless** or **display** version.

## Supported Hardware
- [Astra Machina SL2619 Development Kit](https://www.synaptics.com/products/embedded-processors/sl2610-product-line#devKit)
- [Synaptics Coralboard SL 2GB (SL2619)](https://developers.google.com/coral/products/SL2610-dev-board)


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

### Update the Operating System

The applications in this repository were tested with the following version of the Astra SDK (Yocto-Project Linux):
- **scarthgap_6.12_v2.5.0** - [Relases Page](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.5.0)

From the release page, locate the Out of Box Experience (OOBE) Image for your board and follow the update guide. 

- For the Astra Machina SL2610 Kit:
    - Locate and download the image downloader scrip for **Image for sl2619_oobe_scarthgap**

    - Follow the [Astra Update Guide](https://synaptics-astra.github.io/doc/v/latest/linux/index.html#running-astra-update) to update the eMMC on the board. 


- For the Coralboard *Limited-Edition* from Google IO 2026: 

    - Locate and download the image downloader scrip for **Image for sl2619_coralboard_oobe_scarthgap**


    - Follow the [Astra Update Guide](https://synaptics-astra.github.io/doc/v/latest/linux/index.html#running-astra-update) to update the eMMC 


- For the Coralboard SL 2GB: 
    - Locate and download the image downloader scrip for **Image for sl2619_coralboard_oobe_scarthgap**

    - Follow the [Booting from SD Cards Guide](https://synaptics-astra.github.io/doc/v/latest/linux/index.html#booting-from-spi-and-sd-cards) to generate a bootable SD Card image


### Connect to the SL2610 

Power up the SL2610 kit and open a terminal - using ADB or other method. See hardware setup guide for details.    

### Connect to the network

To enable online example updates and installation of dependencies, it is recommended to connect the kit to the network. See board-specific hardware setup guide for details. 

There are three ways to connect an Astra development kit to the network.
1. Ethernet (Astra Machina only)
2. Network Sharing over USB (Astra SDK 2.3 and later) [Read More](https://developers.google.com/coral/products/SL2610-user-guide#connect_to_a_network_over_usb)
3. WiFi/BT Module (Ampak AP12611_M2 with SYN43711) [Read More](https://developers.google.com/coral/products/SL2610-user-guide#attach_wifibluetooth_module_optional)

### Clone the Repository


Change to the home directory. Always work out of the home directory.

> **⚠️ Warning:** `/home/root` is on a different partition than the root directory (`/`). The root partition has limited space, so cloning the repository or storing models/data outside of `/home/root` can cause the device to run out of space.

```bash
cd /home/root/
```



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

### Getting Updates

To update an existing clone with the latest changes, including the torq-examples submodule:

```bash
git pull
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

Check out the README.md files in each of the example folders. 

- [object_detection](object_detection/README.md) - detect objects with YoloV8
- [image_classification](image_classification/README.md) - classify images with MobileNetV2
- [speech_to_text](speech_to_text/README.md) - capture speech sentences with Moonshine
- [speech_to_text_streaming](speech_to_text_streaming/README.md) - capture speech word-by-word with Moonshine V2
- [gemma_translation](gemma_translate/README.md) - translate text to other languages using Gemma3
- [function_calling](Function_calling/README.md) - control a device with natural language
- [jellectronica](jellectronica/README.md) - turn a video stream into ambient music with Melody RNN


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
