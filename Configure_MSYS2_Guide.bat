@echo off
REM ============================================================================
REM Hướng dẫn cấu hình MSYS2 và cài đặt MinGW-w64 sau khi cài đặt
REM ============================================================================

echo.
echo ======================================================
echo HƯỚNG DẪN CẤU HÌNH SAU KHI CÀI ĐẶT MSYS2
echo ======================================================
echo.

echo Sau khi cài đặt MSYS2 xong, mở "MSYS2 MSYS" từ Start Menu
echo và chạy các lệnh sau:
echo.
echo --- Bước 1: Update cơ sở dữ liệu gói ---
echo.
echo   pacman -Syu
echo.
echo --- Bước 2: Cài đặt các công cụ phát triển ---
echo.
echo   pacman -S --needed base-devel mingw-w64-x86_64-toolchain
echo.
echo   Chọn tất cả các gói (nhấn A và Enter)
echo.
echo --- Bước 3: Cài đặt CMake và các công cụ khác ---
echo.
echo   pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-make
echo.
echo --- Bước 4: Kiểm tra phiên bản GCC ---
echo.
echo   gcc --version
echo.
echo   Nếu thấy phiên bản >= 13.x là OK
echo.
echo --- Bước 5: Thêm vào PATH ---
echo.
echo   export PATH="/c/msys64/mingw64/bin:$PATH"
echo.
echo ======================================================
echo SAU KHI CẤU HÌNH XONG, CHẠY Build_CUDA_Mingw64.bat
echo ======================================================
echo.

pause