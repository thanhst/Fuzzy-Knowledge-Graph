@echo off
setlocal
call "%~dp0Bat run\Run_Full_GPU_Validation.bat" %*
exit /b %errorlevel%
