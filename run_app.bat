@echo off
setlocal enabledelayedexpansion

rem Change to this script's directory
pushd "%~dp0"

rem Detect Python 3.12 at fixed path first, then fall back
set "PY_CMD="
if exist "C:\Python312\python.exe" (
  set "PY_CMD=C:\Python312\python.exe"
) else (
  where py >nul 2>nul
  if %ERRORLEVEL%==0 (
    rem Prefer specific minor versions known to have wheels for SciPy on Windows
    py -3.12 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.12"
    if not defined PY_CMD py -3.11 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.11"
    if not defined PY_CMD py -3.10 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.10"
    if not defined PY_CMD set "PY_CMD=py -3"
  ) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
      set "PY_CMD=python"
    ) else (
      echo [ERROR] Python is not installed or not on PATH. Please install Python 3.10–3.12 or place it at C:\Python312 and try again.
      pause
      exit /b 1
    )
  )
)

rem Create venv if it doesn't exist
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PY_CMD% -m venv .venv
  if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

rem Verify app file exists
if not exist "streamlit_app.py" (
  echo [ERROR] streamlit_app.py not found in %cd%
  pause
  exit /b 1
)

rem Ensure Streamlit is installed
".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if %ERRORLEVEL% neq 0 (
  echo [ERROR] Streamlit is not installed in the virtual environment.
  echo Run update_deps.bat to install dependencies.
  pause
  exit /b 1
)

rem Run the Streamlit app
echo Starting Streamlit app ...
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py

popd
endlocal

