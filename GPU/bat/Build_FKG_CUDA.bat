@echo off
setlocal EnableExtensions

for %%i in ("%~dp0..\..") do set "ROOT=%%~fi\"
set "SRC_DIR=%ROOT%GPU\Source"
set "BUILD_DIR=%SRC_DIR%\build_cuda"
set "PYTHON_META=%ROOT%.fisa_python_path.txt"
set "ALLOW_FALLBACK=0"
if /I "%~1"=="--fallback-cpu" set "ALLOW_FALLBACK=1"

set "PYTHON_EXE="
if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

set "PYBIND11_DIR="
set "PYBIND11_TMP=%TEMP%\fisa_pybind11_dir.txt"
"%PYTHON_EXE%" -c "import pybind11; print(pybind11.get_cmake_dir())" > "%PYBIND11_TMP%" 2>nul
if exist "%PYBIND11_TMP%" (
    set /p PYBIND11_DIR=<"%PYBIND11_TMP%"
    del /q "%PYBIND11_TMP%" >nul 2>nul
)
if not defined PYBIND11_DIR (
    echo [ERROR] Khong tim thay pybind11 cho Python: %PYTHON_EXE%
    echo [ERROR] Cai dat bang lenh: "%PYTHON_EXE%" -m pip install pybind11
    goto :fail
)

set "VCVARS_BAT="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find VC\Auxiliary\Build\vcvars64.bat`) do (
        if not defined VCVARS_BAT set "VCVARS_BAT=%%i"
    )
)
if not defined VCVARS_BAT (
    for %%p in (
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    ) do (
        if not defined VCVARS_BAT if exist %%~p set "VCVARS_BAT=%%~p"
    )
)

set "CUDA_ROOT=%CUDA_PATH%"
if not defined CUDA_ROOT set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

if not defined VCVARS_BAT (
    echo [ERROR] Khong tim thay MSVC vcvars64.bat.
    echo [ERROR] CUDA build tren Windows bat buoc can MSVC + nvcc.
    if "%ALLOW_FALLBACK%"=="1" (
        echo [INFO] Chuyen fallback sang build CPU...
        call "%ROOT%Build_FISA_CPU.bat"
        exit /b %errorlevel%
    )
    exit /b 1
)

if not exist "%CUDA_ROOT%\bin\nvcc.exe" (
    echo [ERROR] Khong tim thay nvcc tai "%CUDA_ROOT%\bin\nvcc.exe"
    if "%ALLOW_FALLBACK%"=="1" (
        echo [INFO] Chuyen fallback sang build CPU...
        call "%ROOT%Build_FISA_CPU.bat"
        exit /b %errorlevel%
    )
    exit /b 1
)

echo ========================================
echo Build FKG Module (CUDA)
echo ========================================
echo Source      : %SRC_DIR%
echo Python      : %PYTHON_EXE%
echo pybind11    : %PYBIND11_DIR%
echo CUDA root   : %CUDA_ROOT%
echo vcvars64    : %VCVARS_BAT%
echo Build folder: %BUILD_DIR%
echo ========================================

if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
> "%PYTHON_META%" echo(%PYTHON_EXE%

call "%VCVARS_BAT%"
if errorlevel 1 goto :fail

cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DUSE_CUDA=ON ^
    -DUSE_GPU=OFF ^
    -Dpybind11_DIR="%PYBIND11_DIR%" ^
    -DPython3_EXECUTABLE="%PYTHON_EXE%" ^
    -DCUDAToolkit_ROOT="%CUDA_ROOT%" ^
    -DFISA_OUTPUT_DIR="%SRC_DIR%"
if errorlevel 1 goto :fail

cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 goto :fail

echo.
echo [OK] Build CUDA thanh cong.
echo Kiem tra file output:
dir /b "%SRC_DIR%\fisa_module*.pyd" 2>nul
exit /b 0

:fail
echo.
echo [ERROR] Build CUDA that bai.
if exist "%PYTHON_META%" del /q "%PYTHON_META%" >nul 2>nul
exit /b 1
