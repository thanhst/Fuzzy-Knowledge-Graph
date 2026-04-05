@echo off
setlocal EnableExtensions

rem =========================================
rem ICTA under5 flow runner (one-click)
rem - allows patients with fewer than 5 images
rem =========================================

for %%I in ("%~dp0..\..") do set "ROOT_DIR=%%~fI"
set "SOURCE_DIR=%ROOT_DIR%\Source"
set "VENV_PY=%ROOT_DIR%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [INFO] .venv not found. Creating virtual environment...
    where py >nul 2>&1
    if errorlevel 1 (
        python -m venv "%ROOT_DIR%\.venv"
    ) else (
        py -3 -m venv "%ROOT_DIR%\.venv"
    )
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv
        goto :fail
    )
)

set "PYTHON_EXE=%VENV_PY%"

rem --- Dependency precheck in .venv only (no system fallback)
"%PYTHON_EXE%" -c "import pandas,numpy,openpyxl,cv2,sklearn,docx,matplotlib" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required Python packages into .venv...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    if errorlevel 1 goto :fail
    "%PYTHON_EXE%" -m pip install pandas numpy openpyxl opencv-python scikit-learn python-docx matplotlib
    if errorlevel 1 (
        echo [ERROR] Failed to install required packages into .venv
        goto :fail
    )
)

if not defined K_TAB set "K_TAB=8"
if not defined K_IMG set "K_IMG=5"
if not defined UNDER5_MAX_IMAGES set "UNDER5_MAX_IMAGES=5"
if not defined UNDER5_IMAGE_SIZE set "UNDER5_IMAGE_SIZE=512"
if not defined FKG_BACKEND set "FKG_BACKEND=gpu"
if not defined FKG_BINS set "FKG_BINS=6"
if not defined FKG_TEST_RATIO set "FKG_TEST_RATIO=0.30"
if not defined FKG_SEED set "FKG_SEED=42"

echo [INFO] ============================================
echo [INFO] ICTA under5 flow started: %date% %time%
echo [INFO] ROOT_DIR          : %ROOT_DIR%
echo [INFO] PYTHON_EXE        : %PYTHON_EXE%
echo [INFO] K_TAB             : %K_TAB%
echo [INFO] K_IMG             : %K_IMG%
echo [INFO] UNDER5_MAX_IMAGES : %UNDER5_MAX_IMAGES%
echo [INFO] UNDER5_IMAGE_SIZE : %UNDER5_IMAGE_SIZE%
echo [INFO] FKG_BACKEND       : %FKG_BACKEND%
echo [INFO] ============================================

cd /d "%ROOT_DIR%" || (
    echo [ERROR] Cannot change directory to %ROOT_DIR%
    goto :fail
)

echo [STEP] 1/6 Process table...
"%PYTHON_EXE%" "%SOURCE_DIR%\Src\Flow\process_table.py"
if errorlevel 1 goto :fail

echo [STEP] 2/6 Process image under5...
"%PYTHON_EXE%" "%SOURCE_DIR%\Src\Flow\process_image_under5.py" ^
  --max-images %UNDER5_MAX_IMAGES% ^
  --image-size %UNDER5_IMAGE_SIZE%
if errorlevel 1 goto :fail

echo [STEP] 3/6 Select table features...
"%PYTHON_EXE%" "%SOURCE_DIR%\Src\Flow\select_table_features.py" --k %K_TAB%
if errorlevel 1 goto :fail

echo [STEP] 4/6 Select image features (under5)...
"%PYTHON_EXE%" "%SOURCE_DIR%\Src\Flow\select_image_features.py" ^
  --input-csv "%SOURCE_DIR%\Data\processing\image_under5\image_features_patient.csv" ^
  --out-dir "%SOURCE_DIR%\Data\processing\image_under5" ^
  --k %K_IMG%
if errorlevel 1 goto :fail

echo [STEP] 5/6 Fusion (under5)...
"%PYTHON_EXE%" "%SOURCE_DIR%\Src\Flow\fusion.py" ^
  --table-csv "%SOURCE_DIR%\Data\processing\table\table_features_selected.csv" ^
  --image-csv "%SOURCE_DIR%\Data\processing\image_under5\image_features_selected.csv" ^
  --out-dir "%SOURCE_DIR%\Data\processing\fusion_under5"
if errorlevel 1 goto :fail

echo [STEP] 6/6 Run FKG flow (under5 fusion)...
"%PYTHON_EXE%" "%SOURCE_DIR%\Src\Flow\run_fkg_gpu_flow.py" ^
  --fusion-csv "%SOURCE_DIR%\Data\processing\fusion_under5\fusion_selected.csv" ^
  --out-dir "%SOURCE_DIR%\Data\result\ICTA_under5" ^
  --backend %FKG_BACKEND% ^
  --bins %FKG_BINS% ^
  --test-ratio %FKG_TEST_RATIO% ^
  --seed %FKG_SEED%
if errorlevel 1 goto :fail

echo [INFO] ============================================
echo [INFO] ICTA under5 flow completed successfully.
echo [INFO] Result folder: %SOURCE_DIR%\Data\result\ICTA_under5
echo [INFO] Finished at: %date% %time%
echo [INFO] ============================================
pause
exit /b 0

:fail
echo [ERROR] ICTA under5 flow failed. Please check the error log above.
echo [INFO] Failed at: %date% %time%
pause
exit /b 1
