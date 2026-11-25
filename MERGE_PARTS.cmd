@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0merge_release_parts.ps1"
set "MERGE_EXIT=%ERRORLEVEL%"
if not "%ANIMALACT_NO_PAUSE%"=="1" pause
exit /b %MERGE_EXIT%
