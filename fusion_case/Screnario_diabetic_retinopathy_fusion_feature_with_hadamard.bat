@echo off
setlocal enabledelayedexpansion

rem --- Logtime
echo [INFO] --- Script started at: %date% %time%

rem --- set workdir
cd ..
cd Source_code

rem --- install code FKG
pip install --find-links=module/Setup_module/CMAKE/wheel/wheelhouse/window fisa_module


rem
echo [INFO] --- Running the Python script...

rem --- Running python by module
python -m main.diabetic_retinopathy.Scenario_diabetic_retinopathy_fusion_feature_with_hadamard

rem --- Check error
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
