@echo off
setlocal
call "%~dp0GPU\bat\Build_FKG_CUDA.bat" --fallback-cpu %*
exit /b %errorlevel%
