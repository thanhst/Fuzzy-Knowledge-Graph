@echo off
setlocal enabledelayedexpansion

rem ============================================================
rem Test Run FKG Full - Batch script to test FKG with real data
rem Uses small dataset for faster execution
rem ============================================================

echo [INFO] ====================================================
echo [INFO] Testing FKG with Real Data
echo [INFO] ====================================================
echo.

rem --- Logtime
echo [INFO] Script started at: %date% %time%

echo.
echo [INFO] --- Step 1: Install fisa_module ---
pip install --find-links=Source_code/module/Setup_module/CMAKE/wheel/wheelhouse/window fisa_module

rem --- check error
if %errorlevel% neq 0 (
    echo [ERROR] --- pip install failed.
    echo [INFO] --- Press any key to exit...
    pause
    exit /b 1
)

echo.
echo [INFO] --- Step 2: Test FKG with existing data ---
echo [INFO] Using small dataset: statistical (fewer features, faster processing)
echo [INFO] Running test_fkg_small.py...

python test_fkg_small.py

rem --- check error
if %errorlevel% neq 0 (
    echo [ERROR] --- FKG execution failed.
    echo [INFO] --- Press any key to exit...
    pause
    exit /b 1
)

rem --- finish
echo.
echo [INFO] ====================================================
echo [SUCCESS] FKG with Real Data Test Passed!
echo [INFO] ====================================================
echo.
echo [INFO] The FKG source code is working properly with real data.
echo.
echo [INFO] Press any key to exit...
pause

rem
echo [INFO] Script finished at: %date% %time%
