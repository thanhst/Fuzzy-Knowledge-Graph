@echo off
setlocal EnableExtensions

for %%i in ("%~dp0..\..") do set "ROOT=%%~fi\"
set "HINT_FILE=%ROOT%.msys2_mingw_bin.txt"
set "CLEAN_ROOT=D:\msys2_clean"

set "PACMAN_EXE="
for %%p in (
    "D:\msys2\usr\bin\pacman.exe"
    "C:\msys64\usr\bin\pacman.exe"
) do (
    if not defined PACMAN_EXE if exist %%~p set "PACMAN_EXE=%%~p"
)

if not defined PACMAN_EXE (
    echo [ERROR] Khong tim thay pacman.exe cua MSYS2.
    echo [ERROR] Can cai MSYS2 truoc, vi du tai D:\msys2 hoac C:\msys64.
    exit /b 1
)

echo ========================================
echo Install / Verify MSYS2 dependencies
echo ========================================
echo pacman: %PACMAN_EXE%
echo ========================================

"%PACMAN_EXE%" -S --needed --noconfirm ^
    mingw-w64-x86_64-gcc ^
    mingw-w64-x86_64-cmake ^
    mingw-w64-x86_64-ninja ^
    mingw-w64-x86_64-python ^
    mingw-w64-x86_64-pybind11
if errorlevel 1 goto :fail

for %%i in ("%PACMAN_EXE%") do set "PACMAN_DIR=%%~dpi"
for %%i in ("%PACMAN_DIR%..\..") do set "MSYS2_ROOT=%%~fi"
set "MINGW_BIN=%MSYS2_ROOT%\mingw64\bin"

if not exist "%MINGW_BIN%\gcc.exe" (
    echo [ERROR] Khong tim thay gcc.exe tai "%MINGW_BIN%".
    goto :fail
)

set "NEED_CLEAN_COPY=0"
if not "%MINGW_BIN: =%"=="%MINGW_BIN%" set "NEED_CLEAN_COPY=1"

set "TMP_CPP=%TEMP%\fisa_msys2_probe.cpp"
set "TMP_OBJ=%TEMP%\fisa_msys2_probe.obj"
> "%TMP_CPP%" echo(int main(){return 0;})
"%MINGW_BIN%\g++.exe" -o "%TMP_OBJ%" -c "%TMP_CPP%" >nul 2>nul
if errorlevel 1 set "NEED_CLEAN_COPY=1"
if exist "%TMP_CPP%" del /q "%TMP_CPP%" >nul 2>nul
if exist "%TMP_OBJ%" del /q "%TMP_OBJ%" >nul 2>nul

if "%NEED_CLEAN_COPY%"=="1" (
    echo [INFO] Toolchain goc khong on dinh hoac duong dan co khoang trang, dang tao ban sao:
    echo        "%MINGW_BIN%" ^> "%CLEAN_ROOT%\mingw64\bin"
    if not exist "%CLEAN_ROOT%" mkdir "%CLEAN_ROOT%"
    robocopy "%MSYS2_ROOT%\mingw64" "%CLEAN_ROOT%\mingw64" /E /NFL /NDL /NJH /NJS /NC /NS
    if errorlevel 8 (
        echo [ERROR] Robocopy that bai.
        goto :fail
    )
    set "MINGW_BIN=%CLEAN_ROOT%\mingw64\bin"
)

if not exist "%MINGW_BIN%\python.exe" (
    echo [ERROR] Khong tim thay python.exe tai "%MINGW_BIN%".
    goto :fail
)

> "%HINT_FILE%" echo(%MINGW_BIN%

echo.
echo [OK] MSYS2 dependencies san sang.
echo [OK] Mingw bin: %MINGW_BIN%
echo [OK] Da ghi hint: %HINT_FILE%
exit /b 0

:fail
echo.
echo [ERROR] Cai dat/kiem tra MSYS2 dependencies that bai.
exit /b 1
