@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "models\distilbert-detector\model.safetensors" goto missing
".venv\Scripts\python.exe" launcher.py %*
if errorlevel 1 pause
exit /b
:missing
echo Run setup.cmd first while connected to the internet.
pause
exit /b 1

