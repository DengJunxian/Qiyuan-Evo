@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0start_competition_demo.ps1" %*
endlocal
