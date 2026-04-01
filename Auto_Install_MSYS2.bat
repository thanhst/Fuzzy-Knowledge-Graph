@echo off
REM ============================================================================
REM CÀI ĐẶT TỰ ĐỘNG MSYS2 VÀ MINGW-W64 MỚI
REM ============================================================================

echo.
echo ========================================
echo CÀI ĐẶT MSYS2 + MINGW-W64 CHO FKG/FIS GPU
echo ========================================
echo.

REM Check if installer exists
if not exist msys2_installer.exe (
    echo LỖI: Khong tim thay msys2_installer.exe
    pause
    exit /b 1
)

echo [1/3] Dang khoi dong trinh cai dat MSYS2...
echo.
echo Vui long hoan thanh cai dat theo huong dan tren man hinh:
echo 1. Chon "Next" 
echo 2. Chon duong dan C:\msys64 (khuyen nghi)
echo 3. Hoan tat cai dat
echo 4. KHONG chon "Run MSYS2 now" - dong cua so lai
echo.
echo Sau khi cai xong, nhan Enter de tiep tuc...
pause

echo.
echo [2/3] Kiem tra MSYS2...
if not exist C:\msys64\msys2_shell.cmd (
    echo LOI: MSYS2 chua duoc cai dat dung!
    echo Vui long cai dat MSYS2 truoc.
    pause
    exit /b 1
)

echo OK! MSYS2 da duoc cai dat.
echo.
echo [3/3] Tao shortcut...
echo.

REM Create a batch to run MSYS2 commands
echo @echo off > install_mingw.bat
echo echo Dang cai dat MinGW-w64... >> install_mingw.bat
echo "C:\msys64\usr\bin\bash.exe" --login -c "pacman -Syu --noconfirm" >> install_mingw.bat
echo "C:\msys64\usr\bin\bash.exe" --login -c "pacman -S --noconfirm --needed base-devel mingw-w64-x86_64-toolchain" >> install_mingw.bat
echo "C:\msys64\usr\bin\bash.exe" --login -c "pacman -S --noconfirm mingw-w64-x86_64-cmake mingw-w64-x86_64-make" >> install_mingw.bat
echo echo. >> install_mingw.bat
echo echo Hoan tat! Kiem tra phiên bản GCC: >> install_mingw.bat
echo "C:\msys64\usr\bin\bash.exe" --login -c "gcc --version" >> install_mingw.bat
echo pause >> install_mingw.bat

echo.
echo Da tao file install_mingw.bat
echo.
echo De tiep tuc, vui long:
echo 1. Mo "C:\msys64\msys2_shell.cmd" (tu File Explorer)
echo 2. Trong cua so MSYS2, copy va chay cac lenh sau:
echo.
echo === COPY TU DAY ===
echo pacman -Syu
echo pacman -S --needed base-devel mingw-w64-x86_64-toolchain
echo pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-make
echo gcc --version
echo === DEN DAY ===
echo.
echo Sau khi hoan thanh (gcc version >= 13.x), chay Build_CUDA_Mingw64.bat
echo.
pause