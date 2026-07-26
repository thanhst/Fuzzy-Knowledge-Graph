@echo off
setlocal EnableExtensions

for %%i in ("%~dp0..") do set "ROOT=%%~fi\"
call "%ROOT%Bat run\Run_Full_GPU_Validation.bat" %*
exit /b %errorlevel%
