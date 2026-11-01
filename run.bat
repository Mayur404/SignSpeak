@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Creating a virtual environment...
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
    echo Installing dependencies. This can take a few minutes...
    "%PYTHON%" -m pip install --upgrade pip
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
)

"%PYTHON%" app.py %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo SignSpeak exited with an error.
pause
exit /b 1
