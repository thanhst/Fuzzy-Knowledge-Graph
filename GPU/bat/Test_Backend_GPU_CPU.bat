@echo off
setlocal EnableExtensions

for %%i in ("%~dp0..\..") do set "ROOT=%%~fi\"

set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=auto"

set "MODULE_DIR=%~2"
if "%MODULE_DIR%"=="" set "MODULE_DIR=auto"

set "CUDA_BIN="
if defined CUDA_PATH set "CUDA_BIN=%CUDA_PATH%\bin"
if not defined CUDA_BIN set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if exist "%CUDA_BIN%\cudart64_12.dll" (
    set "PATH=%CUDA_BIN%;%PATH%"
)

set "PYTHON_EXE="
if exist "%ROOT%.fisa_python_path.txt" (
    for /f "usebackq delims=" %%i in ("%ROOT%.fisa_python_path.txt") do set "PYTHON_EXE=%%i"
)
if not defined PYTHON_EXE if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
)
if not defined PYTHON_EXE if exist "D:\msys2_clean\mingw64\bin\python.exe" (
    set "PYTHON_EXE=D:\msys2_clean\mingw64\bin\python.exe"
)
if not defined PYTHON_EXE (
    set "PYTHON_EXE=python"
)

echo ========================================
echo Test backend CPU/GPU
echo ========================================
echo Python    : %PYTHON_EXE%
echo Backend   : %BACKEND%
echo Module dir: %MODULE_DIR%
echo ========================================

"%PYTHON_EXE%" -u "%ROOT%test_backend_gpu_cpu.py" --backend "%BACKEND%" --module-dir "%MODULE_DIR%"
if errorlevel 1 (
    echo [ERROR] Backend test that bai.
    exit /b 1
)

echo [OK] Backend test thanh cong.
exit /b 0
