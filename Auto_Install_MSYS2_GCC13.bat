@echo off
REM ============================================================================
REM Auto-install MSYS2 with GCC 13/14 for building FKG/FIS with CUDA
REM ============================================================================

setlocal enabledelayedexpansion

echo ============================================
echo Installing MSYS2 with GCC 13/14
echo ============================================

REM Check if MSYS2 already installed
if exist "C:\msys64\usr\bin\pacman.exe" (
    echo MSYS2 already installed at C:\msys64
    goto :install_gcc
)

REM Download MSYS2
echo.
echo Downloading MSYS2 installer...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/msys2/msys2-installer/releases/download/2024-01-13/msys2-x86_64-20240113.exe' -OutFile 'msys2_installer.exe'"

if not exist msys2_installer.exe (
    echo ERROR: Failed to download MSYS2
    pause
    exit /b 1
)

echo Installing MSYS2...
start /wait msys2_installer.exe /D=C:\msys64 /S

REM Wait for installation
timeout /t 30 /nobreak > nul

if not exist "C:\msys64\usr\bin\pacman.exe" (
    echo ERROR: MSYS2 installation failed
    pause
    exit /b 1
)

echo MSYS2 installed successfully

:install_gcc
echo.
echo Installing GCC 13...

REM Run pacman to install mingw-w64-x86_64-gcc
C:\msys64\usr\bin\bash.exe -lc "pacman -Sy --noconfirm mingw-w64-x86_64-gcc"

echo.
echo ============================================
echo MSYS2 + GCC installation complete!
echo ============================================
echo.
echo Next steps:
echo 1. Run: C:\msys64\msys2_shell.cmd
echo 2. In the shell, run: pacman -Syu
echo 3. Then run the build script
echo.

pause