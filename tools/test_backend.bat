@echo off
setlocal EnableExtensions

for %%i in ("%~dp0..") do set "ROOT=%%~fi\"
call "%ROOT%Bat run\Test_Backend_GPU_CPU.bat" %*
exit /b %errorlevel%
