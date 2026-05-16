@echo off
REM ========================================================================
REM   Driver Intent Monitoring System -- launcher (Windows)
REM
REM   Interactive menu:
REM     1. Scan + pick driver-facing camera
REM     2. Optional camera/gaze calibration (uses the cam from step 1)
REM     3. Pick setup: two-camera real-road OR MetaDrive simulator
REM     4. (Two-cam only) pick scene cam
REM     5. (Sim only) auto-pilot or manual control
REM
REM   Power-user shortcut: pass any args to bypass the menu, e.g.
REM     RUN.bat --mode two-cam --driver 0 --scene 1
REM     RUN.bat --mode metadrive --manual --quick
REM     RUN.bat --mode calibrate
REM ========================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"
title Driver Intent Monitor

REM Pick a Python interpreter: prefer the bundled venv, fall back to system.
set "PY="
if exist ".venv311\Scripts\python.exe" (
    set "PY=.venv311\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if !ERRORLEVEL!==0 set "PY=python"
)

if "%PY%"=="" (
    echo.
    echo [ERROR] No Python interpreter found.
    echo   Expected: .venv311\Scripts\python.exe
    echo   or a 'python' on PATH.
    echo.
    pause
    exit /b 1
)

REM ── Power-user passthrough: if any args were given, skip the menu ──
if not "%~1"=="" (
    "%PY%" launcher.py %*
    set "RC=!ERRORLEVEL!"
    goto end
)

echo.
echo ============================================================
echo   Driver Intent Monitoring System
echo ============================================================
echo.

REM ─── Step 1: Scan + pick driver-facing camera ──────────────────
echo  Step 1 -- Driver-facing camera
echo  ------------------------------
echo  Scanning attached cameras...
"%PY%" launcher.py --list-cameras
if errorlevel 1 (
    echo.
    echo  [ERROR] No working cameras found. Connect a camera and re-run.
    pause
    exit /b 1
)
echo.
echo  Look at the resolutions above. The DRIVER camera should be the
echo  one pointed at your face.
echo.
set /p DRIVER_CAM=Driver-cam index [default 0, press Enter to accept]:
if "!DRIVER_CAM!"=="" set "DRIVER_CAM=0"
echo  Driver cam = !DRIVER_CAM!
echo.

REM ─── Step 2: Camera / gaze calibration ─────────────────────────
echo  Step 2 -- Camera and gaze calibration
echo  -------------------------------------
echo  Calibration improves gaze tracking accuracy. Run it the
echo  first time you use the system, or whenever lighting,
echo  seating, or camera position has changed.
echo  Calibration will use driver camera !DRIVER_CAM!.
echo.
choice /C YN /N /M "Run camera/gaze calibration first? [Y/N]: "
if errorlevel 2 (
    echo  Skipping calibration.
) else (
    echo.
    echo  Launching calibration on driver cam !DRIVER_CAM!...
    "%PY%" launcher.py --mode calibrate --driver !DRIVER_CAM! --quick
    if errorlevel 1 (
        echo.
        echo  ======================================================
        echo  [WARNING] Calibration exited with a non-zero code.
        echo  Scroll up to read the error message above.
        echo  ======================================================
        echo  Press any key to continue with existing calibration
        echo  ^(if any^) or close this window to abort.
        pause
    )
)
echo.

REM ─── Step 3: Setup mode ────────────────────────────────────────
echo  Step 3 -- Pick setup
echo  --------------------
echo    [1] Two-camera ^(driver cam + real road / scene cam^)
echo    [2] Simulator  ^(driver cam + MetaDrive scene^)
echo.
choice /C 12 /N /M "Pick setup [1/2]: "
if errorlevel 2 (
    set "SETUP=sim"
    echo  Simulator setup selected.
) else (
    set "SETUP=two-cam"
    echo  Two-camera setup selected.
)
echo.

REM ─── Step 4: Scene cam (two-cam only) ──────────────────────────
if "!SETUP!"=="two-cam" (
    echo  Step 4 -- Scene camera
    echo  ----------------------
    set /p SCENE_CAM=Scene-cam index [default 1, press Enter to accept]:
    if "!SCENE_CAM!"=="" set "SCENE_CAM=1"
    echo  Scene cam  = !SCENE_CAM!
    echo.
)

REM ─── Step 5: Drive mode (sim only) ─────────────────────────────
set "MANUAL_ARG="
if "!SETUP!"=="sim" (
    echo  Step 5 -- Pick how the MetaDrive vehicle is controlled
    echo  ------------------------------------------------------
    echo    [1] Auto-pilot   ^(simulator drives itself; observe only^)
    echo    [2] Manual       ^(you drive with keyboard: W A S D^)
    echo.
    choice /C 12 /N /M "Pick mode [1/2]: "
    if errorlevel 2 (
        set "MANUAL_ARG=--manual"
        echo  Manual mode selected.
    ) else (
        echo  Auto-pilot mode selected.
    )
    echo.
)

REM ─── Launch ────────────────────────────────────────────────────
echo  Launching inference...
echo  ----------------------
echo  Model       : auto-detect ensemble ^(models\weak_v2_fold*.pth^)
if "!SETUP!"=="two-cam" (
    echo  Setup       : two-camera real-road
    echo  Driver cam  : !DRIVER_CAM!
    echo  Scene cam   : !SCENE_CAM!
    echo.
    "%PY%" launcher.py --mode two-cam --driver !DRIVER_CAM! --scene !SCENE_CAM! --quick
) else (
    echo  Setup       : MetaDrive simulator
    echo  Driver cam  : !DRIVER_CAM!
    if defined MANUAL_ARG (
        echo  Control     : manual keyboard
    ) else (
        echo  Control     : auto-pilot
    )
    echo.
    "%PY%" launcher.py --mode metadrive --driver !DRIVER_CAM! !MANUAL_ARG! --quick
)
set "RC=!ERRORLEVEL!"

:end
echo.
echo ------------------------------------------------------------
if "%RC%"=="0" (
    echo Run finished. Output bundle is in:  %CD%\runs\
    echo Look at:  violations.csv  warnings.csv  summary.md
) else (
    echo Run exited with code %RC%.
)
echo ------------------------------------------------------------
echo.
pause
endlocal
