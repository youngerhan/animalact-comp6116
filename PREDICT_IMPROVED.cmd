@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"
set "PATH=%~dp0runtime;%~dp0runtime\Library\bin;%~dp0runtime\Scripts;%PATH%"
pushd "%~dp0AnimalAct-SlowFast"
"%~dp0runtime\python.exe" predict_improved.py ..\AnimalAct\videos\test\AAWJEBGU.mp4 --start-frame 0 --end-frame 35
set "RUN_EXIT=%ERRORLEVEL%"
popd
if not "%ANIMALACT_NO_PAUSE%"=="1" pause
exit /b %RUN_EXIT%
