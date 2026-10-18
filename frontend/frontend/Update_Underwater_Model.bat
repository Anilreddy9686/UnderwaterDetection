@echo off
setlocal
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%~dp0Ultralytics"
if not exist "venv\Scripts\python.exe" (
  echo Project virtual environment was not found: %~dp0venv\Scripts\python.exe
  pause
  exit /b 1
)
if not exist "%USERPROFILE%\Downloads\archive.zip" (
  echo Dataset ZIP not found: %USERPROFILE%\Downloads\archive.zip
  echo Download the underwater dataset as archive.zip or edit this file to set its location.
  pause
  exit /b 1
)
echo Preparing the combined original and plastic training data...
"venv\Scripts\python.exe" prepare_underwater_dataset.py --archive "%USERPROFILE%\Downloads\archive.zip"
if errorlevel 1 (
  echo Dataset preparation failed.
  pause
  exit /b 1
)
echo Training the expanded model. This can take a long time on a CPU.
"venv\Scripts\python.exe" train_expanded_model.py
if errorlevel 1 echo Training did not finish. The original model remains unchanged.
pause
