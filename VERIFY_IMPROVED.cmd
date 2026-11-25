@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"
set "PATH=%~dp0runtime;%~dp0runtime\Library\bin;%~dp0runtime\Scripts;%PATH%"
pushd "%~dp0AnimalAct-SlowFast"
"%~dp0runtime\python.exe" report_improved.py
set "RUN_EXIT=%ERRORLEVEL%"
popd
if not "%ANIMALACT_NO_PAUSE%"=="1" pause
exit /b %RUN_EXIT%
