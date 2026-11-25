@echo off
setlocal
chcp 65001 >nul
set "PYTHONHOME=%~dp0runtime"
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "MPLCONFIGDIR=%~dp0_runtime_cache\matplotlib"
set "PATH=%~dp0runtime;%~dp0runtime\Libraryin;%~dp0runtime\Scripts;%PATH%"
cd /d "%~dp0"
"%~dp0runtime\python.exe" "%~dp0run_bundle.py" verify %*
set "BUNDLE_EXIT=%ERRORLEVEL%"
echo.
echo Finished with exit code %BUNDLE_EXIT%.
if not "%ANIMALACT_NO_PAUSE%"=="1" pause
exit /b %BUNDLE_EXIT%
