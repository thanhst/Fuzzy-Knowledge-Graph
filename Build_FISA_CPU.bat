@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "SRC_DIR=%ROOT%Source"
set "BUILD_DIR=%SRC_DIR%\build_cpu"
set "PYTHON_META=%ROOT%.fisa_python_path.txt"
set "MINGW_HINT_FILE=%ROOT%.msys2_mingw_bin.txt"

set "MINGW_BIN="
if exist "%MINGW_HINT_FILE%" (
    for /f "usebackq delims=" %%i in ("%MINGW_HINT_FILE%") do set "MINGW_BIN=%%i"
)
if defined MINGW_BIN if not exist "%MINGW_BIN%\gcc.exe" set "MINGW_BIN="
if not defined MINGW_BIN if exist "D:\msys2_clean\mingw64\bin\gcc.exe" set "MINGW_BIN=D:\msys2_clean\mingw64\bin"
if not defined MINGW_BIN if exist "%ROOT%msys2\mingw64\bin\gcc.exe" set "MINGW_BIN=%ROOT%msys2\mingw64\bin"
if not defined MINGW_BIN if exist "D:\msys2\mingw64\bin\gcc.exe" set "MINGW_BIN=D:\msys2\mingw64\bin"
if not defined MINGW_BIN if exist "C:\msys64\mingw64\bin\gcc.exe" set "MINGW_BIN=C:\msys64\mingw64\bin"
if not defined MINGW_BIN if exist "D:\Program file\mingw64\bin\gcc.exe" set "MINGW_BIN=D:\Program file\mingw64\bin"

if defined MINGW_BIN (
    if not "%MINGW_BIN: =%"=="%MINGW_BIN%" (
        for %%p in ("%MINGW_BIN%\..\..") do set "MSYS2_ROOT=%%~fp"
        set "MSYS2_LINK=%TEMP%\fisa_msys2_link"
        if exist "%MSYS2_LINK%" rmdir "%MSYS2_LINK%" >nul 2>nul
        cmd /c mklink /J "%MSYS2_LINK%" "%MSYS2_ROOT%" >nul 2>nul
        if exist "%MSYS2_LINK%\mingw64\bin\gcc.exe" (
            echo [INFO] Tao junction tam thoi de tranh duong dan co khoang trang.
            set "MINGW_BIN=%MSYS2_LINK%\mingw64\bin"
            set "MINGW_LINK_CREATED=1"
        ) else (
            echo [WARNING] Bo qua MinGW vi duong dan co khoang trang: %MINGW_BIN%
            set "MINGW_BIN="
        )
    )
)

set "VCVARS_BAT="
for %%p in (
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) do (
    if not defined VCVARS_BAT if exist %%~p set "VCVARS_BAT=%%~p"
)

set "PYTHON_EXE="
if defined VCVARS_BAT (
    if exist "%ROOT%.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
    ) else (
        set "PYTHON_EXE=python"
    )
) else (
    if defined MINGW_BIN (
        if exist "%MINGW_BIN%\python.exe" (
            set "PYTHON_EXE=%MINGW_BIN%\python.exe"
        ) else (
            if exist "%ROOT%.venv\Scripts\python.exe" (
                set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
            ) else (
                set "PYTHON_EXE=python"
            )
        )
    ) else (
        if exist "%ROOT%.venv\Scripts\python.exe" (
            set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
        ) else (
            set "PYTHON_EXE=python"
        )
    )
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
    echo Neu dang dung MSYS2: pacman -S mingw-w64-x86_64-pybind11
    echo Neu dang dung venv: "%PYTHON_EXE%" -m pip install pybind11
    exit /b 1
)

echo ========================================
echo Build FISA Module (CPU)
echo ========================================
echo Python      : %PYTHON_EXE%
echo pybind11    : %PYBIND11_DIR%
if defined VCVARS_BAT (
    echo Toolchain   : MSVC
    echo vcvars64    : %VCVARS_BAT%
) else (
    if defined MINGW_BIN (
        echo Toolchain   : MinGW
        echo MinGW bin   : %MINGW_BIN%
    ) else (
        echo Toolchain   : MinGW ^(PATH^)
    )
)
echo Build folder: %BUILD_DIR%
echo ========================================

if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
> "%PYTHON_META%" echo(%PYTHON_EXE%

if defined VCVARS_BAT (
    call "%VCVARS_BAT%"
    if errorlevel 1 goto :fail

    cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DUSE_CUDA=OFF ^
        -DUSE_GPU=OFF ^
        -Dpybind11_DIR="%PYBIND11_DIR%" ^
        -DPython3_EXECUTABLE="%PYTHON_EXE%"
) else (
    if defined MINGW_BIN (
        set "PATH=%MINGW_BIN%;%PATH%"
        cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" -G "MinGW Makefiles" ^
            -DCMAKE_BUILD_TYPE=Release ^
            -DUSE_CUDA=OFF ^
            -DUSE_GPU=OFF ^
            -Dpybind11_DIR="%PYBIND11_DIR%" ^
            -DPython3_EXECUTABLE="%PYTHON_EXE%" ^
            -DCMAKE_C_COMPILER="%MINGW_BIN%\gcc.exe" ^
            -DCMAKE_CXX_COMPILER="%MINGW_BIN%\g++.exe" ^
            -DCMAKE_MAKE_PROGRAM="%MINGW_BIN%\mingw32-make.exe"
    ) else (
        cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" -G "MinGW Makefiles" ^
            -DCMAKE_BUILD_TYPE=Release ^
            -DUSE_CUDA=OFF ^
            -DUSE_GPU=OFF ^
            -Dpybind11_DIR="%PYBIND11_DIR%" ^
            -DPython3_EXECUTABLE="%PYTHON_EXE%"
    )
)
if errorlevel 1 goto :fail

cmake --build "%BUILD_DIR%" --config Release -j 4
if errorlevel 1 goto :fail

echo.
echo [OK] Build CPU thanh cong.
echo Kiem tra file output:
dir /b "%SRC_DIR%\fisa_module*.pyd" 2>nul
echo.
if not defined VCVARS_BAT (
    echo Neu gap loi link Py_* voi MinGW, hay cai Visual Studio Build Tools
    echo de build bang MSVC.
)
if defined MINGW_LINK_CREATED if exist "%MSYS2_LINK%" rmdir "%MSYS2_LINK%" >nul 2>nul
exit /b 0

:fail
echo.
echo [ERROR] Build CPU that bai.
if exist "%PYTHON_META%" del /q "%PYTHON_META%" >nul 2>nul
if defined MINGW_LINK_CREATED if exist "%MSYS2_LINK%" rmdir "%MSYS2_LINK%" >nul 2>nul
exit /b 1
