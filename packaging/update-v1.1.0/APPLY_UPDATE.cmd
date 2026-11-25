@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
py -3.11 -c "import sys" >nul 2>&1
if errorlevel 1 goto try_python
py -3.11 "%~dp0apply_update.py" %*
goto done

:try_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if errorlevel 1 goto bundle_python
python "%~dp0apply_update.py" %*
goto done

:bundle_python
echo 未找到 Python 3.11，将使用旧完整包附带的 Python。
set /p "UPDATE_TARGET=请输入旧完整包根目录（包含 runtime）: "
set "UPDATE_TARGET=%UPDATE_TARGET:"=%"
if not exist "%UPDATE_TARGET%\runtime\python.exe" goto missing_runtime
set "PYTHONHOME=%UPDATE_TARGET%\runtime"
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PATH=%UPDATE_TARGET%\runtime;%UPDATE_TARGET%\runtime\Library\bin;%UPDATE_TARGET%\runtime\Scripts;%PATH%"
"%UPDATE_TARGET%\runtime\python.exe" "%~dp0apply_update.py" --target "%UPDATE_TARGET%" %*
goto done

:missing_runtime
echo 未找到该目录的 runtime\python.exe。请先完整解压 v1.0.0 复现包。
exit /b 2

:done
set "UPDATE_EXIT=%ERRORLEVEL%"
echo.
echo Finished with exit code %UPDATE_EXIT%.
if not "%ANIMALACT_NO_PAUSE%"=="1" pause
exit /b %UPDATE_EXIT%
