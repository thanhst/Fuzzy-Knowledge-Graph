@echo off
setlocal
call "%~dp0Bat run\Build_FKG_CUDA.bat" --fallback-cpu %*
exit /b %errorlevel%
