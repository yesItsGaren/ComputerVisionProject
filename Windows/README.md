# Windows Setup — OpenCV Color Detection

## Requirements

- Python 3.11 (already installed at `C:\Users\arabk\AppData\Local\Programs\Python\Python311\`)
- A camera connected via USB

## Install dependencies

Open PowerShell and run:

```powershell
C:\Users\arabk\AppData\Local\Programs\Python\Python311\python.exe -m pip install opencv-python numpy
```

> **Why this specific pip?** You must call `pip` through the same Python executable you use to run the scripts, otherwise packages install into the wrong environment and you get `ModuleNotFoundError` again.

Verify the install worked:

```powershell
C:\Users\arabk\AppData\Local\Programs\Python\Python311\python.exe -c "import cv2; print(cv2.__version__)"
```

---

## Cameras

| Folder   | Camera         | Resolution       | Backend     |
|----------|----------------|------------------|-------------|
| Logitech | Logitech C925e | 1920×1080        | CAP_DSHOW   |
| IMX415   | Sony IMX415    | 3840×2160 (4K)   | CAP_DSHOW   |

Both scripts auto-detect the camera index by probing for the expected resolution — you do not need to hardcode an index.

---

## Workflow

### Step 1 — Calibrate HSV values

Run the calibration script for your camera:

```powershell
# Logitech
C:\Users\arabk\AppData\Local\Programs\Python\Python311\python.exe Windows\Logitech\calibrationWindows.py

# IMX415
C:\Users\arabk\AppData\Local\Programs\Python\Python311\python.exe Windows\IMX415\calibrationWindows.py
```

- Adjust the six trackbars (H/S/V min and max) until only your target object appears white in the mask panel.
- Press **S** to print the calibrated values to the terminal.
- Press **Q** to quit.

### Step 2 — Paste values into the detection script

Open the corresponding `colorDetectionWindows.py` and replace the placeholder values:

```python
# --- Paste your calibrated values here ---
lower = np.array([H_MIN, S_MIN, V_MIN])
upper = np.array([H_MAX, S_MAX, V_MAX])
```

### Step 3 — Run detection

```powershell
# Logitech
C:\Users\arabk\AppData\Local\Programs\Python\Python311\python.exe Windows\Logitech\colorodDetectionWindows.py

# IMX415
C:\Users\arabk\AppData\Local\Programs\Python\Python311\python.exe Windows\IMX415\colorDetectionWindows.py
```

Press **Q** to quit.

---

## What the detection script does

1. Captures frames from the camera.
2. Converts each frame from BGR to HSV color space.
3. Applies a color mask using your calibrated HSV range.
4. Runs morphological open/close to remove noise and fill holes in the mask.
5. Finds contours and draws a bounding box + centroid dot on each detected object.
6. Displays a live FPS counter (green ≥ 20 fps, orange ≥ 10, red < 10).

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Wrong Python / pip | Install via the exact Python path shown above |
| `IMX415 not found` | Camera not recognized at 4K | Try unplugging and replugging the USB cable; check Device Manager |
| Black screen / frozen frame | Buffered old frames | Already handled — `CAP_PROP_BUFFERSIZE = 1` is set |
| Detection fires on everything | HSV range too wide | Re-run calibration and narrow the trackbars |
| Detection misses the object | HSV range too narrow | Re-run calibration and widen slightly |
