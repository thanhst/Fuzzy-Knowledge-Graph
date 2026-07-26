@echo off
setlocal EnableExtensions

for %%i in ("%~dp0..") do set "ROOT=%%~fi\"

set "PYTHON_EXE="
if exist "%ROOT%.fisa_python_path.txt" (
    for /f "usebackq delims=" %%i in ("%ROOT%.fisa_python_path.txt") do set "PYTHON_EXE=%%i"
)
if not defined PYTHON_EXE if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
)
if not defined PYTHON_EXE set "PYTHON_EXE=python"

"%PYTHON_EXE%" "%ROOT%runners\icta\run_icta_flow.py" %*
exit /b %errorlevel%
