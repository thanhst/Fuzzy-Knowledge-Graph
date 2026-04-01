@echo off
REM ============================================================================
REM CÀI ĐẶT MSYS2 VÀ BUILD FKG/FIS VỚI GPU
REM ============================================================================

echo.
echo ========================================
echo Bước 1: Cài đặt MSYS2
echo ========================================
echo.

if not exist msys2_installer.exe (
    echo LỖI: Khong tim thay msys2_installer.exe
    pause
    exit /b 1
)

echo Dang khoi dong trinh cai dat...
start msys2_installer.exe

echo.
echo SAU KHI CAi DAT XONG, VUI LONG:
echo 1. Dong trinh cai dat
echo 2. Mo "MSYS2 MSYS" tu Start Menu
echo 3. Copy va chay cac lenh sau trong cua so MSYS2:
echo.
echo === COPY TU DAY ===
echo pacman -Syu
echo pacman -S --needed base-devel mingw-w64-x86_64-toolchain
echo pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-make
echo gcc --version
echo === DEN DAY ===
echo.
echo Sau khi hoan thanh, nhap phim bat ky de thoat...
pause