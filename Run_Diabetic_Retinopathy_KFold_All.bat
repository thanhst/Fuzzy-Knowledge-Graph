@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not defined PYTHON_EXE (
  if exist ".venv_deep_baselines\Scripts\python.exe" (
    set "PYTHON_EXE=.venv_deep_baselines\Scripts\python.exe"
  ) else (
    set "PYTHON_EXE=python"
  )
)

if not defined RUN_ID (
  for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_ID=%%I"
)

if not defined DEVICE set "DEVICE=auto"
if not defined RESNET_ARCH set "RESNET_ARCH=resnet50"
if not defined FIS_BACKEND set "FIS_BACKEND=cpu"
if not defined FKG_BACKEND set "FKG_BACKEND=auto"
if not defined FKGS_RAN set "FKGS_RAN=15 20"
if not defined FKGS_EPSILON set "FKGS_EPSILON=0.2 0.3"
if not defined FKGS_TURNS set "FKGS_TURNS=1"
if not defined FKGS_WORKERS set "FKGS_WORKERS=4"

set "DEEP_RESULTS=ROOT_DATA\train_test_selection\deep_baselines\kfold_rerun_%RUN_ID%"
set "FKGS_OUTPUT=data\Dataset_diabetic\KFold_feature_selection_rerun_%RUN_ID%"
set "FKGS_REPORT=data\result\KFold_feature_selection_rerun_%RUN_ID%"
set "FKGS_REPORT_ABS=Source_code\%FKGS_REPORT%"
set "COMPARISON_STEM=result\diabetic_retinopathy_model_comparison_kfold_rerun_%RUN_ID%"

echo ============================================================
echo Diabetic Retinopathy KFold full rerun
echo RUN_ID=%RUN_ID%
echo PYTHON_EXE=%PYTHON_EXE%
echo DEVICE=%DEVICE%
echo RESNET_ARCH=%RESNET_ARCH%
echo FKG_BACKEND=%FKG_BACKEND%
echo FKGS_RAN=%FKGS_RAN%
echo FKGS_EPSILON=%FKGS_EPSILON%
echo FKGS_WORKERS=%FKGS_WORKERS%
echo.
echo Deep baseline output: %DEEP_RESULTS%
echo FKGS output: %FKGS_OUTPUT%
echo FKGS report: %FKGS_REPORT_ABS%
echo Comparison: %COMPARISON_STEM%.csv / .md
echo ============================================================
echo.

echo [1/4] Creating patient-aware train/test and train KFold manifests...
"%PYTHON_EXE%" "Source_code\main\diabetic_retinopathy\create_root_data_image_train_test_split.py" ^
  --root-data "ROOT_DATA" ^
  --image-dir "ROOT_DATA\fundus_photos_224" ^
  --materialize none ^
  --path-mode relative ^
  --overwrite
if errorlevel 1 goto fail

echo.
echo [2/4] Running deep baselines on patient-aware KFold splits...
"%PYTHON_EXE%" "Source_code\main\diabetic_retinopathy\run_deep_multimodal_baselines.py" ^
  --split-root "ROOT_DATA\train_test_selection" ^
  --tabular-csv "Source_code\data\Dataset_diabetic\data_process.csv" ^
  --results-dir "%DEEP_RESULTS%" ^
  --models all ^
  --epochs 10 ^
  --batch-size 16 ^
  --device "%DEVICE%" ^
  --resnet-arch "%RESNET_ARCH%"
if errorlevel 1 goto fail

echo.
echo [3/4] Running FIS + FKGS + native FKG for image, table, and fusion KFold splits...
"%PYTHON_EXE%" "Source_code\main\diabetic_retinopathy\Preprocess_kfold_feature_selection.py" ^
  --modalities image table fusion ^
  --run-fkgs ^
  --run-fkg ^
  --fis-engine native ^
  --native-backend "%FIS_BACKEND%" ^
  --fkg-backend "%FKG_BACKEND%" ^
  --ran %FKGS_RAN% ^
  --e %FKGS_EPSILON% ^
  --fkgs-turns "%FKGS_TURNS%" ^
  --fkgs-workers "%FKGS_WORKERS%" ^
  --output-root "%FKGS_OUTPUT%" ^
  --report-root "%FKGS_REPORT%"
if errorlevel 1 goto fail

echo.
echo [4/4] Building comparison table...
"%PYTHON_EXE%" "Source_code\main\diabetic_retinopathy\collect_kfold_model_comparison.py" ^
  --deep-summary "%DEEP_RESULTS%\summary.csv" ^
  --deep-config "%DEEP_RESULTS%\config.json" ^
  --fkgs-summary "%FKGS_REPORT_ABS%\kfold_fkgs_mean_std_summary.csv" ^
  --fkgs-tables "%FKGS_REPORT_ABS%\kfold_fkgs_tables.csv" ^
  --fkg-summary "%FKGS_REPORT_ABS%\kfold_modality_mean_std_summary.csv" ^
  --output-stem "%COMPARISON_STEM%"
if errorlevel 1 goto fail

echo.
echo [DONE] Full KFold rerun completed.
echo Deep baseline output: %DEEP_RESULTS%
echo FKGS report: %FKGS_REPORT_ABS%
echo Comparison CSV: %COMPARISON_STEM%.csv
echo Comparison Markdown: %COMPARISON_STEM%.md
endlocal & exit /b 0

:fail
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [FAILED] Full KFold rerun stopped with exit code %EXIT_CODE%.
echo Check the console output above and any console.log written under %DEEP_RESULTS%.
endlocal & exit /b %EXIT_CODE%
