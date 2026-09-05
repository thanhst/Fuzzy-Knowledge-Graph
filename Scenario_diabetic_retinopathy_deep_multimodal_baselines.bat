@echo off
setlocal

if exist ".venv_deep_baselines\Scripts\python.exe" (
  set PYTHON_EXE=.venv_deep_baselines\Scripts\python.exe
) else (
  set PYTHON_EXE=python
)

"%PYTHON_EXE%" "Source_code\main\diabetic_retinopathy\run_deep_multimodal_baselines.py" ^
  --split-root "ROOT_DATA\train_test_selection" ^
  --tabular-csv "Source_code\data\Dataset_diabetic\data_process.csv" ^
  --models all ^
  --epochs 10 ^
  --batch-size 16 ^
  --run-final-test ^
  %*

set EXIT_CODE=%ERRORLEVEL%
endlocal & exit /b %EXIT_CODE%
