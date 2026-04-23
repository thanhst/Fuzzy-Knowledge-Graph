@echo off
setlocal
call "%~dp0GPU\bat\Build_FKG_CUDA_MSYS2.bat" --fallback-cpu %*
exit /b %errorlevel%
