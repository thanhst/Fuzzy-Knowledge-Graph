@echo off
setlocal EnableExtensions

for %%i in ("%~dp0..") do set "ROOT=%%~fi\"

if "%~1"=="" (
    call "%ROOT%Bat run\Build_FKG_CUDA.bat" --fallback-cpu
) else (
    call "%ROOT%Bat run\Build_FKG_CUDA.bat" %*
)
exit /b %errorlevel%
