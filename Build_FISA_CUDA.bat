@echo off
setlocal
echo [INFO] Script goc da chuyen sang Bat run\Build_FKG_CUDA.bat
echo [INFO] Dang goi pipeline moi ...
call "%~dp0Bat run\Build_FKG_CUDA.bat" --fallback-cpu %*
exit /b %errorlevel%
