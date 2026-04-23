@echo off
setlocal enabledelayedexpansion

rem --- Logtime
echo [INFO] --- Script started at: %date% %time%

rem --- set workdir
cd /d "%~dp0"

rem --- Running the Python script
echo [INFO] --- Running Symile-MIMIC Fusion Pipeline (FIS-FKG-FKGS)...

rem --- Running python by module
python -m Source_code.main.DuyHoang3ModalFusion.run.Fusion

rem --- check error
if %errorlevel% neq 0 (
    echo [ERROR] --- Python script execution failed. Check the error above.
    echo [INFO] --- Press any key to exit...
    pause
    exit /b
)

rem --- finish
echo [INFO] --- Python script executed successfully.

rem
echo [INFO] --- Press any key to exit...
pause

rem
echo [INFO] --- Script finished at: %date% %time%
