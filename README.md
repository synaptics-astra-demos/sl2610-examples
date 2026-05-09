# Synaptics Astra SL2610 Series AI Examples

This repository provides AI example applications for the **Synaptics Astra SL2610** series. Currently, there are only a few simple **computer vision** examples. However, check back for future updates to include examples for **speech processing and large language models (LLMs)**. Follow the instructions below to set up your environment and run various AI examples in few minutes.

The examples in this repository are designed to work with Astra SL2610 processor using Astra Machina Dev Kit. Examples leverage the NPU.

## Learn more about Synaptics Astra by visiting:

- [Astra](https://www.synaptics.com/products/embedded-processors) – Explore the Astra AI platform.
- [Astra Machina](https://www.synaptics.com/products/embedded-processors/astra-machina-foundation-series) – Discover our powerful development kit.
- [AI Developer Zone](https://developer.synaptics.com/) – Find step-by-step tutorials and resources.

## Setting up Astra Machina Board
For instructions on how to set up Astra Machina board , see the  [Setting up the hardware](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html)  guide.


## Torq Compiler & Runtime

The Torq compiler is based on the MLIR framework and IREE runtime. The examples use the Torq compiler to optimize models to run efficiently on the Torq NPU. See Torq documentation for details.  
    
- [Torq Documentation](https://synaptics-torq.github.io/torq-compiler/v/latest)


## 🔧 Installation
 
### Connect to the SL2610 

Power up the Astra Machina SL2610 board and open a terminal. 

### Clone the Repository

Clone the repository using the following command:

```bash
git clone https://github.com/synaptics-astra-demos/sl2610-examples
```
Navigate to the Repository Directory:

```bash
cd sl2610-examples
```

### Setup Python Environment

To get started, set up your Python environment. This step ensures all required dependencies are installed and isolated within a virtual environment:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

Install dependencies

If online
```bash
pip install -r requirements.txt
```

If offline
```bash
pip install --no-index --find-links=./wheelhouse -r requirements.txt
```

## Object Detection

Follow the steps in /object_detection/README.md to see how to perform object detection using YoloV8 on a single image. 

## Image Classification

Follow the steps in /image_classification/README.md to see how to perform image classification using MobileNetV2 on a single image. 

## Speech To Text (Moonshine)

Follow the steps in /speech_to_text/README.md to see how to perform speed-to-text function using Moonshine Model with a microphone. 