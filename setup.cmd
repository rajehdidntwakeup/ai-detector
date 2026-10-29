@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" setup_model.py
if errorlevel 1 goto failed
echo.
echo Setup finished. Double-click start.cmd to use the app offline.
pause
exit /b 0
:failed
echo.
echo Setup failed. Check your internet connection and that Python 3.10-3.13 is installed.
pause
exit /b 1

