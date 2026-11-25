@echo off
setlocal
set "PYTHONHOME=%~dp0runtime"
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONIOENCODING=utf-8"
set "PATH=%~dp0runtime;%~dp0runtime\Libraryin;%PATH%"
"%~dp0runtime\python.exe" "%~dp0check_files.py"
set "BUNDLE_EXIT=%ERRORLEVEL%"
if not "%ANIMALACT_NO_PAUSE%"=="1" pause
exit /b %BUNDLE_EXIT%
