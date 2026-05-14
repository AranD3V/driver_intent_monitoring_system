@echo off
REM ========================================================================
REM   Driver Intent Monitoring System -- single-click launcher (Windows)
REM
REM   Double-click:           launches the MetaDrive free-roam demo with
REM                            traffic lights, sim labelling and violation
REM                            recording. No prompts, no menu. Press q
REM                            or Esc to stop.
REM
REM   Run from a terminal:    any CLI args are passed through to
REM                            launcher.py, eg
REM                              RUN.bat --mode two-cam
REM                              RUN.bat --mode metadrive            (autopilot)
REM                              python launcher.py                  (full menu)
REM ========================================================================

setlocal

REM Always run from the directory this .bat lives in, regardless of where
REM the user double-clicked it from.
cd /d "%~dp0"

title Driver Intent Monitor

REM Pick a Python interpreter: prefer the bundled venv, fall back to system.
set "PY="
if exist ".venv311\Scripts\python.exe" (
    set "PY=.venv311\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 set "PY=python"
)

if "%PY%"=="" (
    echo.
    echo [ERROR] No Python interpreter found.
    echo   Expected: .venv311\Scripts\python.exe
    echo   or a 'python' on PATH.
    echo.
    echo Run:  python -m venv .venv311
    echo       .venv311\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Driver Intent Monitoring System
echo   Interpreter: %PY%
echo ------------------------------------------------------------
echo   Free-roam MetaDrive sim (keyboard: W/A/S/D or arrows)
echo   - Traffic lights cycle at every intersection
echo   - Lane labels: broken / solid / yellow / white / edge
echo   - Action labels: accel / brake / coast / steer / lane-change
echo   - Violations recorded to violations.csv:
echo       overspeed, harsh accel/brake, wrong-side,
echo       off-road, crossed-solid, collisions, ran-red-light
echo   - Run keeps going through crashes; press q or Esc to stop
echo ============================================================
echo.

REM No CLI args -> launch the one-click MetaDrive demo (manual control,
REM no prompts). Otherwise forward args verbatim so power users can pick
REM any mode they want.
if "%~1"=="" (
    "%PY%" launcher.py --mode metadrive --manual --quick
) else (
    "%PY%" launcher.py %*
)
set "RC=%ERRORLEVEL%"

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
