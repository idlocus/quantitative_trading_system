@echo off
cd /d "D:\python_work\Trae\quantitative_trading_system"

schtasks /query /tn "AStockScheduler" > nul 2>&1
if %errorlevel%==0 (
    schtasks /delete /tn "AStockScheduler" /f
)

schtasks /create /tn "AStockScheduler" /tr "python \"D:\python_work\Trae\quantitative_trading_system\jobs\scheduler_app.py\"" /sc onlogon /rl LIMITED /f

if %errorlevel%==0 (
    echo SUCCESS: Task created
    echo.
    echo Task Details:
    schtasks /query /tn "AStockScheduler" /fo LIST /v
) else (
    echo ERROR: Failed to create task
)
