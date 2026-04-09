@echo off
chcp 65001 > nul
echo ==========================================
echo   A股调度器 - 开机自启动设置
echo ============================================

cd /d "%~dp0"

REM 检查是否已有任务
schtasks /query /tn "AStockScheduler" > nul 2>&1
if %errorlevel%==0 (
    echo [INFO] 任务已存在，正在删除旧任务...
    schtasks /delete /tn "AStockScheduler" /f
)

REM 获取Python路径
for /f "delims=" %%i in ('where python 2^>nul') do set PYTHON_PATH=%%i
if not defined PYTHON_PATH (
    set PYTHON_PATH=python
)

echo [INFO] 创建开机自启动任务...
schtasks /create /tn "AStockScheduler" ^
    /tr "\"%PYTHON_PATH%\" \"%cd%\jobs\scheduler_app.py\"" ^
    /sc onlogon ^
    /rl LIMITED ^
    /f

if %errorlevel%==0 (
    echo [SUCCESS] 任务创建成功！
    echo.
    echo 任务详情:
    schtasks /query /tn "AStockScheduler" /v | findstr "Status Run time"
) else (
    echo [ERROR] 任务创建失败！
)

echo.
pause
