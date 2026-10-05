@echo off
rem Double-click to start: checks packages (asks before installing), then opens the photo manager.
rem ASCII only on purpose: cmd misreads non-ASCII batch files. Chinese messages come from start.py.
cd /d "%~dp0"
set "PY="
python --version >nul 2>nul && set "PY=python"
if not defined PY (py -3 --version >nul 2>nul && set "PY=py -3")
if not defined PY (
    echo Python 3.10+ was not found.
    echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
    echo then double-click this file again.
    pause
    exit /b 1
)
%PY% start.py %*
if errorlevel 1 pause
