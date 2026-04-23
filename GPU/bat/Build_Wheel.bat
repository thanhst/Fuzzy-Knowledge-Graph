@echo off
setlocal EnableExtensions

for %%i in ("%~dp0..\..") do set "ROOT=%%~fi\"
set "SOURCE_DIR=%ROOT%Source"

set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=cpu"

set "PYTHON_EXE="
if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

if /I "%BACKEND%"=="gpu" (
    set "USE_CUDA=ON"
    set "USE_GPU=OFF"
    set "FISA_LOCAL_VERSION=cu128"
    set "OUTDIR=%ROOT%dist\wheels\gpu"
) else (
    set "USE_CUDA=OFF"
    set "USE_GPU=OFF"
    set "FISA_LOCAL_VERSION=cpu"
    set "OUTDIR=%ROOT%dist\wheels\cpu"
)

if not exist "%OUTDIR%" mkdir "%OUTDIR%"
del /q "%OUTDIR%\fisa_module-*.whl" >nul 2>nul

echo ========================================
echo Build wheel fisa_module
echo ========================================
echo Python  : %PYTHON_EXE%
echo Backend : %BACKEND%
echo USE_CUDA: %USE_CUDA%
echo Version : %FISA_LOCAL_VERSION%
echo OUTDIR  : %OUTDIR%
echo ========================================

"%PYTHON_EXE%" -m pip install --upgrade build
if errorlevel 1 goto :fail

set "USE_CUDA=%USE_CUDA%"
set "USE_GPU=%USE_GPU%"
set "FISA_LOCAL_VERSION=%FISA_LOCAL_VERSION%"
"%PYTHON_EXE%" -m build --wheel --outdir "%OUTDIR%" "%SOURCE_DIR%"
if errorlevel 1 goto :fail

echo.
echo [OK] Wheel build xong.
dir /b "%OUTDIR%\*.whl" 2>nul
exit /b 0

:fail
echo.
echo [ERROR] Build wheel that bai.
exit /b 1
