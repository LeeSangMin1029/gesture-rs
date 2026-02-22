@echo off
cd /d "%~dp0"
set ORT_DYLIB_PATH=%~dp0onnxruntime.dll
target\release\gesture-control.exe
if errorlevel 1 (
    echo.
    echo [ERROR] Program exited with error code %errorlevel%
)
pause
