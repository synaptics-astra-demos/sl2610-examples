# Image Capture and Scene Description App

Visual scene description demonstrating image understanding

(Currently a placeholder)

## 🔧 Installation
 
### Connect the display

    7" Waveshare Touchscreen panel

### Clone the Repository

Clone the repository and its torq-examples submodule using the following command:

```bash
git clone --recurse-submodules https://github.com/synaptics-astra-demos/sl2610-examples.git
```
Navigate to the Repository Directory:

```bash
cd sl2610-examples
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

### Setup Python Environment

To get started, set up your Python environment. This step ensures all required dependencies are installed and isolated within a virtual environment:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

Install general dependencies

```bash
pip install -r requirements.txt
```


Install specific dependencies for this app

```bash
cd scene_description
pip install -r requirements.txt
```

## Configure for Portrait Mode


1.  Edit /etc/xdg/weston/weston.ini:

    a. In the [core] section, comment out this line 
    
    ```
    [core]
    #shell=/usr/lib/syna-desktop-shell.so
    ```

    b. Comment out the [shell] section. Replace with this single line

    ```       
    [shell]
    panel-position=none
    ```

    b. Update the [output] section 

    ```
    [output]
    name=DPI-1
    mode=800x480@60.0        # Keep it 800x480 even for portrait
    transform=rotate-90      # or rotate-270 depending on physical orientation
    ```

    2.  Restart Weston.

    ```
    systemctl restart weston
    ```

    2.  Edit `app_camera.py`
    
    ```
    ORIENTATION = "portrait" 
    ```



## Usage

Start the app

```bash
python app_camera.py
```
