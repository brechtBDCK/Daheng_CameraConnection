## VIRAL Camera Connection

Python utilities for controlling Daheng cameras from WSL. The repo contains:

1. `main.py` – interactive viewer for previewing, saving frames, and recording video streams.
2. `multi_exp_image.py` – scripted capture pipeline that records multiple exposures and fuses them into an HDR output.

### Features
- Works with Daheng cameras via `gxipy`
- Interactive controls (save frame, toggle recording, quit)
- HDR workflow using OpenCV’s Debevec + Drago operators
- Ready-to-run instructions for WSL + USB/IP setup

---

## Prerequisites
- Windows 11 with WSL2 (Ubuntu recommended)
- Daheng Galaxy SDK for Linux
- Python 3.10+ with `gxipy`, `opencv-python`, `numpy`
- USB/IP tooling (`usbipd-win`) to pass the camera through to WSL

---

## Install the Daheng SDK inside WSL
WSL is its own Linux environment, so Windows `.dll` files are not usable. Install the Linux SDK inside WSL to provide the `libgxiapi.so` shared object.

1. Download the Linux SDK (`Galaxy_Linux-x86_Gige-U3_*.run` or `.tar.gz`) from the Daheng Imaging Download Center.
2. Copy it into WSL (replace the username as needed):
   ```bash
   cp /mnt/c/Users/<username>/Downloads/Galaxy_Linux*.run ~/
   cd ~
   ```
3. Make the installer executable:
   ```bash
   chmod +x Galaxy_Linux-x86_Gige-U3_*.run
   ```
4. Run it with sudo and follow the prompts (default path is `/opt/DxImageCard/Galaxy_SDK/`):
   ```bash
   sudo ./Galaxy_Linux-x86_Gige-U3_*.run
   ```
5. Let Linux know where to find `libgxiapi.so`:
   ```bash
   export LD_LIBRARY_PATH=/opt/DxImageCard/Galaxy_SDK/lib:$LD_LIBRARY_PATH
   echo 'export LD_LIBRARY_PATH=/opt/DxImageCard/Galaxy_SDK/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

---

## Pass the USB camera to WSL
Run these PowerShell commands as Administrator on Windows:
```powershell
usbipd list
usbipd bind --busid <BUSID> --force
usbipd attach --wsl --busid <BUSID>
```
Then in WSL verify visibility with:
```bash
lsusb
```

---

## Project Setup
1. Create and activate a virtual environment (optional but recommended).
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt   # or manually install gxipy, opencv-python, numpy
   ```
3. Confirm the camera is detected:
   ```bash
   python - <<'PY'
   import gxipy as gx
   dm = gx.DeviceManager()
   print("Cameras:", dm.update_device_list()[0])
   PY
   ```

---

## Usage
### Interactive viewer (`main.py`)
```bash
python main.py
```
Controls:
- `s` – save current frame to `output/`
- `r` – toggle AVI recording
- `q` – quit and close the camera

Customize capture settings (exposure, gain, frame size, save path) at the top of the script.

### HDR capture pipeline (`multi_exp_image.py`)
```bash
python multi_exp_image.py
```
This script:
1. Captures a bracketed exposure series (tweak `exposure_times` in the script).
2. Writes images to `output_images/`.
3. Merges them into `output_images/hdr_image.png`.

---

## Troubleshooting
- **`Cannot find libgxiapi.so`** – ensure the SDK is installed in WSL and `LD_LIBRARY_PATH` includes its `lib` directory.
- **`NameError: 'dll'`** – consequence of the missing library above; fix the library path.
- **Camera not detected** – reattach with `usbipd`, confirm with `lsusb`, then re-run the Python script.

Happy capturing!
