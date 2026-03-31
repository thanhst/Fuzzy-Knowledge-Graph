@echo off
setlocal EnableExtensions

echo [INFO] Dang cai/kiem tra dependency bang MSYS2...
call "%~dp0Install_MSYS2_Dependencies.bat"
if errorlevel 1 exit /b 1

echo [INFO] Tiep tuc build CUDA.
echo [INFO] Luu y: tren Windows, CUDA van can MSVC (khong thay the bang MinGW).
call "%~dp0Build_FKG_CUDA.bat" %*
exit /b %errorlevel%
