@echo off
title Stop Quant System
setlocal
echo Stopping Quant Strategy System...

REM Safety-first stop (audit 2026-11-19 IV-f rewrite). The PID file records:
REM   pid        : the webapp process id
REM   start_unix : the process creation time (unix seconds, from GetProcessTimes)
REM   entry      : the project entry path (webapp\app.py)
REM We kill only when ALL of the following hold, and the kill happens in the
REM SAME PowerShell invocation as the verification (no re-read of the record
REM in between -- the old flow re-read the PID file after verifying, so a
REM swapped record made us kill an unverified process):
REM   1. process exists and its image is python
REM   2. record has a creation time AND it matches the live process (+/-2s)
REM      -> missing start_unix = incomplete identity = refuse (keep process)
REM   3. its command line contains the recorded entry path
REM The record file is located relative to THIS script (%~dp0), not the
REM caller's working directory. On identity-check failure the record is KEPT
REM for manual investigation; it is only removed when the process is confirmed
REM gone (stale) or after a verified kill.

set "RC=0"
set "PIDFILE=%~dp0runtime\webapp.pid"
if not exist "%PIDFILE%" (
    echo No PID record found ^(runtime\webapp.pid missing^).
    echo This app is likely not running; nothing was killed.
    goto :done
)

powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $rec = Get-Content -LiteralPath $env:PIDFILE -Raw | ConvertFrom-Json; $p = Get-Process -Id $rec.pid -ErrorAction SilentlyContinue; if (-not $p) { exit 2 }; if ($p.ProcessName -notmatch 'python') { exit 3 }; if ($null -eq $rec.start_unix) { exit 4 }; $recStart = [DateTimeOffset]::FromUnixTimeMilliseconds([long]([double]$rec.start_unix * 1000)).LocalDateTime; if ([Math]::Abs(($p.StartTime - $recStart).TotalSeconds) -gt 2) { exit 4 }; $cl = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $rec.pid)).CommandLine; if (!$rec.entry -or $rec.entry -ne [IO.Path]::GetFullPath((Join-Path (Split-Path (Split-Path $env:PIDFILE)) 'webapp/app.py'))) { exit 5 }; if ($cl -notmatch [regex]::Escape($rec.entry)) { exit 5 }; $p.Kill(); $p.WaitForExit(); exit 0" >nul 2>&1
set RC=%ERRORLEVEL%

if "%RC%"=="0" (
    echo Identity verified (python + creation time + entry path^). Killed in the same operation.
    goto :done
)
if "%RC%"=="2" (
    echo Recorded PID is not running -- stale record, not killing.
    goto :done
)
if "%RC%"=="3" (
    echo Recorded PID is not a python process -- record KEPT for manual check.
    goto :done
)
if "%RC%"=="4" (
    echo PID exists but creation time does not match the record
    echo ^(reused PID or incomplete identity^) -- refusing to kill; record KEPT.
    goto :done
)
if "%RC%"=="5" (
    echo PID is python but its command line is not this project -- record KEPT.
    goto :done
)
echo Identity check failed ^(code %RC%^) -- not killing; record KEPT.

:done
echo.
if "%RC%"=="2" exit /b 0
exit /b %RC%
