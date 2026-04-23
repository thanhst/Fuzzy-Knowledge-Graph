@echo off
setlocal EnableExtensions

for %%i in ("%~dp0..") do set "ROOT=%%~fi\"

set "PYTHON_EXE="
if exist "%ROOT%.fisa_python_path.txt" (
    for /f "usebackq delims=" %%i in ("%ROOT%.fisa_python_path.txt") do set "PYTHON_EXE=%%i"
)
if not defined PYTHON_EXE if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
)
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo [INFO] Step 1/4 - Build CUDA
call "%~dp0Build_FKG_CUDA.bat" --fallback-cpu
if errorlevel 1 exit /b 1

echo [INFO] Step 2/4 - Backend smoke test
call "%~dp0Test_Backend_GPU_CPU.bat" auto source
if errorlevel 1 exit /b 1

echo [INFO] Step 3/4 - Matrix consistency test
"%PYTHON_EXE%" -u "%ROOT%Source\tests\test_fkg_matrix_consistency.py"
if errorlevel 1 exit /b 1

echo [INFO] Step 4/4 - Python/CPU/GPU benchmark
call "%~dp0Test_FKG_Python_vs_CPP_CUDA.bat" auto source
if errorlevel 1 exit /b 1

echo [OK] Full GPU validation passed.
exit /b 0
