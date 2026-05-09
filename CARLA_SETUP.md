# CARLA INSTALLATION GUIDE

> **READ THIS BEFORE YOU TOUCH ANYTHING.**
> Getting CARLA wrong wastes hours. Follow every step in order.

---

## STEP 0 — SYSTEM REQUIREMENTS

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / Ubuntu 18.04 | Windows 11 / Ubuntu 20.04 |
| GPU | NVIDIA GTX 970 (4 GB VRAM) | NVIDIA RTX 2070+ (8 GB VRAM) |
| RAM | 8 GB | 16 GB |
| Disk | 30 GB free | 50 GB free (maps are huge) |
| Python | 3.7 | **3.8** (most stable with CARLA egg) |

> **DO NOT USE PYTHON 3.11+.** The CARLA Python API egg is not compiled for it
> and will silently fail or throw cryptic import errors.

---

## STEP 1 — DOWNLOAD CARLA SERVER

Go to the official CARLA GitHub releases page:

```
https://github.com/carla-simulator/carla/releases
```

Pick a release. **CARLA 0.9.15** is the latest stable as of early 2025.

Download **both** of these files for your OS:

```
CARLA_0.9.15.zip          ← the simulator server
AdditionalMaps_0.9.15.zip ← extra town maps (optional but useful)
```

> **DO NOT** download the "source build" unless you are a CARLA developer.
> The prebuilt package is all you need.

---

## STEP 2 — EXTRACT THE SERVER

### Windows

```
C:\CARLA\CARLA_0.9.15\
```

Extract `AdditionalMaps` into the same folder if you downloaded it.
The folder structure should look like:

```
C:\CARLA\CARLA_0.9.15\
    CarlaUE4.exe       ← this is the server
    CarlaUE4\
    PythonAPI\
        carla\
        examples\
```

### Linux

```bash
mkdir -p ~/carla
cd ~/carla
tar -xzf CARLA_0.9.15.tar.gz
```

---

## STEP 3 — PYTHON VERSION CHECK

**STOP.** Before installing anything:

```bash
python --version
```

You need **3.7 or 3.8**. If you have 3.9+, create a dedicated environment:

```bash
# With conda
conda create -n carla python=3.8
conda activate carla

# With venv (if you have Python 3.8 installed side-by-side)
py -3.8 -m venv venv_carla
venv_carla\Scripts\activate   # Windows
source venv_carla/bin/activate # Linux
```

---

## STEP 4 — INSTALL THE CARLA PYTHON PACKAGE

You have two options. **Use option A unless it fails.**

### Option A — pip (easiest, CARLA 0.9.12+)

```bash
pip install carla==0.9.15
```

> The pip version must **exactly match** your server version.
> Mismatched versions will connect but crash unpredictably mid-session.

Verify it worked:

```bash
python -c "import carla; print(carla.__version__)"
# Expected: 0.9.15
```

### Option B — install from the .egg file (fallback)

If pip fails or you need a version not on PyPI:

```bash
# Windows — find the egg for your Python version
cd C:\CARLA\CARLA_0.9.15\PythonAPI\carla\dist\
dir  # look for something like carla-0.9.15-cp38-cp38-win_amd64.egg

# Install it
easy_install carla-0.9.15-cp38-cp38-win_amd64.egg
```

```bash
# Linux
cd ~/carla/PythonAPI/carla/dist/
easy_install carla-0.9.15-cp38-cp38-linux-x86_64.egg
```

> If `easy_install` is not found: `pip install setuptools` first.

---

## STEP 5 — INSTALL THE REST OF THIS PROJECT'S DEPS

```bash
pip install -r requirements.txt
```

---

## STEP 6 — START THE CARLA SERVER

### Windows

Double-click `CarlaUE4.exe`, **OR** launch from terminal for lower VRAM usage:

```cmd
cd C:\CARLA\CARLA_0.9.15
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600
```

### Linux

```bash
cd ~/carla
./CarlaUE4.sh -quality-level=Low
```

> Wait until you see the CARLA world render in the window before continuing.
> The server takes 20–40 seconds to fully start.

For headless / remote servers (no display):

```bash
./CarlaUE4.sh -RenderOffScreen
```

---

## STEP 7 — VERIFY THE SERVER IS REACHABLE

```bash
python - <<'EOF'
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
print("Connected! Map:", world.get_map().name)
EOF
```

Expected output:

```
Connected! Map: Carla/Maps/Town10HD
```

If you get a timeout → server is not running or firewall is blocking port 2000.

---

## STEP 8 — RUN WITH THIS PROJECT

```bash
# Driver cam = webcam index 0, scene = CARLA on localhost
python inference.py --driver 0 --carla-host localhost --carla-port 2000
```

Optional flags:

```bash
--carla-port 2000    # change if you run CARLA on a non-default port
--output out.mp4     # record session to video
--model weights.pt   # load trained intent model
```

---

## COMMON ERRORS

### `RuntimeError: time-out of 10000ms while waiting for the simulator`
The CARLA server is not running, or the port is wrong.
Check that `CarlaUE4.exe` / `CarlaUE4.sh` is open and fully loaded.

### `ImportError: No module named 'carla'`
The carla package is not installed in the active Python environment.
Run `pip install carla==0.9.15` inside the correct virtualenv.

### `carla.libcarla.Exception: Trying to create a actor in an unknown map`
You connected before the map finished loading.
Add a few extra seconds of wait after starting the server.

### Version mismatch crash mid-session
Your `pip install carla` version != server version.
They must be identical. Check with `python -c "import carla; print(carla.__version__)"`.

### GPU out of memory on startup
Launch the server with `-quality-level=Low` (see Step 6).

### `easy_install` not found
```bash
pip install setuptools
```

---

## QUICK REFERENCE

```bash
# Start server (Windows, low quality)
C:\CARLA\CARLA_0.9.15\CarlaUE4.exe -quality-level=Low -windowed

# Start server (Linux, headless)
~/carla/CarlaUE4.sh -RenderOffScreen

# Verify connection
python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(10); print(c.get_world().get_map().name)"

# Run project in CARLA mode
python inference.py --driver 0 --carla-host localhost
```
