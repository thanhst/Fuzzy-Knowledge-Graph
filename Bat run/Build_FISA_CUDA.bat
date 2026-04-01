@echo off
setlocal
call "%~dp0Build_FKG_CUDA.bat" %*
exit /b %errorlevel%
