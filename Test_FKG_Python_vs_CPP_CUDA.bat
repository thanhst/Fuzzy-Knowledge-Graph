@echo off
setlocal
call "%~dp0Bat run\Test_FKG_Python_vs_CPP_CUDA.bat" %*
exit /b %errorlevel%
