@echo off
REM ========================================================================
REM   Driver Intent Monitoring System -- single-click launcher (Windows)
REM   Double-click this file to start the interactive setup menu.
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
echo   Using interpreter: %PY%
echo ============================================================
echo.

"%PY%" launcher.py %*
set "RC=%ERRORLEVEL%"

echo.
echo ------------------------------------------------------------
if "%RC%"=="0" (
    echo Run finished. Output bundle is in:  %CD%\runs\
) else (
    echo Run exited with code %RC%.
)
echo ------------------------------------------------------------
echo.
pause
endlocal
