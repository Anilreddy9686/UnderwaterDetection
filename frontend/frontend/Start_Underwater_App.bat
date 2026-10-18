@echo off
setlocal
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%~dp0Ultralytics"
if not exist "venv\Scripts\python.exe" (
  echo Project virtual environment was not found.
  echo Check that the venv folder exists in this project.
  pause
  exit /b 1
)
if not exist uploads mkdir uploads
if not exist outputs mkdir outputs
start "" cmd /c "timeout /t 5 /nobreak >nul & start http://127.0.0.1:5000"
"venv\Scripts\python.exe" app.py
pause
