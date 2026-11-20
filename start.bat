@echo off
title Quant Strategy System
cd /d %~dp0

echo =============================================
echo   Quant Strategy System
echo   Close this window to stop server
echo =============================================
echo.

REM Open browser after 3 seconds (background helper, uses default browser)
start /b cmd /c "timeout /t 3 /nobreak >nul & start "" http://localhost:5000"

REM Run server in foreground (Ctrl+C or close window = stop)
".venv\Scripts\python.exe" "%~dp0webapp\app.py"

echo.
echo Server stopped.
pause >nul
