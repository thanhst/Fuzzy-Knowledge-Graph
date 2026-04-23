@echo off
setlocal
echo [INFO] Script goc da chuyen sang Bat run\Test_Backend_GPU_CPU.bat
call "%~dp0Bat run\Test_Backend_GPU_CPU.bat" %*
exit /b %errorlevel%
