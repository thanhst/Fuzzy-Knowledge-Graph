@echo off
setlocal EnableExtensions

for %%i in ("%~dp0..") do set "ROOT=%%~fi\"

set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=auto"

set "MODULE_DIR=%~2"
if "%MODULE_DIR%"=="" set "MODULE_DIR=source"

set "PYTHON_EXE="
if exist "%ROOT%.fisa_python_path.txt" (
    for /f "usebackq delims=" %%i in ("%ROOT%.fisa_python_path.txt") do set "PYTHON_EXE=%%i"
)
if not defined PYTHON_EXE if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
)
if not defined PYTHON_EXE (
    set "PYTHON_EXE=python"
)

set "OUT_JSON=%ROOT%result\benchmark\fkg_python_cpp_cuda.json"

echo ========================================
echo Benchmark: Python vs C++ CPU vs CUDA
echo ========================================
echo Python    : %PYTHON_EXE%
echo Backend   : %BACKEND%
echo Module dir: %MODULE_DIR%
echo Output    : %OUT_JSON%
echo ========================================

"%PYTHON_EXE%" -u "%ROOT%Source\tests\test_fkg_python_vs_cpp_cuda.py" ^
    --backend "%BACKEND%" ^
    --module-dir "%MODULE_DIR%" ^
    --samples 420 ^
    --features 6 ^
    --classes 3 ^
    --out-json "%OUT_JSON%"

if errorlevel 1 (
    echo [ERROR] Benchmark script that bai.
    exit /b 1
)

echo [OK] Benchmark script thanh cong.
exit /b 0
