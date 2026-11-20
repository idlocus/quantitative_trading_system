# 注册每日自动交易计划任务（Windows 任务计划程序）
# 用法: 以管理员身份运行  powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1
# 效果: 每个交易日 16:30 自动运行 main.py daily（数据→信号→Paper成交→日志→净值）
# 卸载: Unregister-ScheduledTask -TaskName "QuantDaily" -Confirm:$false

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) { throw "未找到 venv Python: $python" }

$action  = New-ScheduledTaskAction -Execute $python -Argument "main.py daily" -WorkingDirectory $root
# 工作日 16:30（收盘后数据基本就绪；节假日运行会因无新数据而零交易，无害）
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At ([datetime]'16:30')
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName "QuantDaily" -Action $action -Trigger $trigger `
    -Settings $settings -Description "量化系统每日自动流程（Paper）" -Force

Write-Host "已注册计划任务 QuantDaily：工作日 16:30 自动运行（输出见 logs\daily_task.log）"
Write-Host "手动测试: Start-ScheduledTask -TaskName QuantDaily"
