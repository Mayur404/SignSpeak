@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%PYTHON%" goto :run

rem ---- First run: create .venv and install dependencies -------------------
rem (Kept outside a ( ) block: %PYTHON% inside a block expands before the
rem  "set" in that block runs, so pip would have used the old, missing path.)
echo Creating a virtual environment...
where py >nul 2>nul
if errorlevel 1 (python -m venv .venv) else (py -3 -m venv .venv)
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" goto :error
echo Installing dependencies. This can take a few minutes...
"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

:run
rem ---- Port and scheme come from .env (WEB_PORT, WEB_HTTPS) ----------------
set "PORT=18473"
set "SCHEME=https"
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if /i "%%A"=="WEB_PORT" if not "%%B"=="" set "PORT=%%B"
        if /i "%%A"=="WEB_HTTPS" (
            if /i "%%B"=="0" set "SCHEME=http"
            if /i "%%B"=="false" set "SCHEME=http"
            if /i "%%B"=="no" set "SCHEME=http"
            if /i "%%B"=="off" set "SCHEME=http"
        )
    )
)
echo %* | findstr /i /c:"--http" >nul && set "SCHEME=http"

echo.
echo Starting SignSpeak Web UI
echo   Open %SCHEME%://localhost:%PORT%   (change WEB_PORT in .env to use another port)
echo   Press Ctrl+C to stop. Run only one copy at a time: two copies fight over the webcam.
echo.
"%PYTHON%" app.py %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo SignSpeak exited with an error.
pause
exit /b 1
