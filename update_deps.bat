@echo off
setlocal enabledelayedexpansion

rem Change to this script's directory
pushd "%~dp0"
set "FAILED=0"

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

rem If an existing venv was created with Python 3.13, recreate it with a compatible version
if exist ".venv\pyvenv.cfg" (
  for /f "tokens=3 delims= " %%v in ('findstr /B /C:"version =" ".venv\pyvenv.cfg"') do set VENV_PY_VER=%%v
  if "!VENV_PY_VER:~0,4!"=="3.13" (
    echo Detected existing venv with Python !VENV_PY_VER!; recreating venv with %PY_CMD% ...
    rmdir /s /q .venv
  )
)

rem Create venv if it doesn't exist
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv using %PY_CMD% ...
  %PY_CMD% -m venv .venv
  if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    set "FAILED=1"
    goto END
  )
)

rem Try to make sure pip/setuptools are present and up-to-date in the venv (Python 3.11+)
%PY_CMD% -m venv .venv --upgrade-deps >nul 2>nul

rem Ensure pip is available inside the venv
echo Ensuring pip is available in the virtual environment ...
".venv\Scripts\python.exe" -m ensurepip --upgrade --default-pip >nul 2>nul

".venv\Scripts\python.exe" -m pip --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
  echo pip not found via ensurepip, attempting bootstrap with get-pip.py ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'; exit 0 } catch { exit 1 }" >nul 2>nul || (
    echo PowerShell download failed. Trying curl ...
    curl -L -o get-pip.py https://bootstrap.pypa.io/get-pip.py >nul 2>nul
  )
  if not exist "get-pip.py" (
    echo [ERROR] Failed to download get-pip.py. Check your internet connection or proxy settings.
    set "FAILED=1"
    goto END
  )
  ".venv\Scripts\python.exe" get-pip.py
  set BOOTSTRAP_RC=%ERRORLEVEL%
  del /q get-pip.py >nul 2>nul
  if %BOOTSTRAP_RC% neq 0 (
    echo [ERROR] Failed to bootstrap pip inside the virtual environment.
    set "FAILED=1"
    goto END
  )
)

echo Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
  echo [ERROR] pip upgrade failed.
  set "FAILED=1"
  goto END
)

if exist "requirements.txt" (
  echo Installing/updating dependencies from requirements.txt ...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install one or more dependencies from requirements.txt
    set "FAILED=1"
    goto END
  )
) else (
  echo [ERROR] requirements.txt not found. Nothing to install.
  set "FAILED=1"
  goto END
)

echo Dependencies updated successfully.

:END
if "%FAILED%"=="1" (
  echo.
  echo An error occurred during dependency setup.
  echo Please review the messages above.
)
echo.
pause
popd
endlocal

