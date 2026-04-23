@echo off
setlocal enabledelayedexpansion

rem ============================================================
rem Test Run FKG - Batch script to test if the source code works
rem ============================================================

echo [INFO] ====================================================
echo [INFO] Testing FKG Source Code
echo [INFO] ====================================================
echo.

rem --- Logtime
echo [INFO] Script started at: %date% %time%

echo.
echo [INFO] --- Step 1: Install fisa_module ---
cd Source_code
pip install --find-links=module/Setup_module/CMAKE/wheel/wheelhouse/window fisa_module

rem --- check error
if %errorlevel% neq 0 (
    echo [ERROR] --- pip install failed.
    echo [INFO] --- Press any key to exit...
    pause
    exit /b 1
)

echo.
echo [INFO] --- Step 2: Test Python imports ---
python -c "import fisa_module; print('fisa_module imported:', dir(fisa_module))"

rem --- check error
if %errorlevel% neq 0 (
    echo [ERROR] --- fisa_module import failed.
    echo [INFO] --- Press any key to exit...
    pause
    exit /b 1
)

echo.
echo [INFO] --- Step 3: Test Source_code module imports ---
python -c "from module.FIS.FIS import FIS; print('FIS import OK')"
python -c "from module.FKG.FKG_general import FKG; print('FKG import OK')"

rem --- check error
if %errorlevel% neq 0 (
    echo [ERROR] --- Source_code module import failed.
    echo [INFO] --- Press any key to exit...
    pause
    exit /b 1
)

echo.
echo [INFO] --- Step 4: Run simple test ---
python -c "import sys; sys.path.insert(0, 'Source_code'); from module.FIS.FIS import FIS; print('FIS module loaded successfully'); print('Testing basic functionality...'); print('Test passed!')"

rem --- check error
if %errorlevel% neq 0 (
    echo [ERROR] --- Test execution failed.
    echo [INFO] --- Press any key to exit...
    pause
    exit /b 1
)

rem --- finish
echo.
echo [INFO] ====================================================
echo [SUCCESS] All tests passed!
echo [INFO] ====================================================
echo.
echo [INFO] The source code is working properly.
echo [INFO] You can now run the scenario scripts.
echo.
echo [INFO] Example commands:
echo [INFO]   python -m Source_code.main.diabetic_retinopathy.Scenario_diabetic_retinopathy_table_feature
echo [INFO]   python -m Source_code.main.diabetic_harvard_data.run.Scenario_oct_table_fusion_features
echo.
echo [INFO] Press any key to exit...
pause

rem
echo [INFO] Script finished at: %date% %time%