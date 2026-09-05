@echo off
setlocal

if exist ".venv_deep_baselines\Scripts\python.exe" (
  set PYTHON_EXE=.venv_deep_baselines\Scripts\python.exe
) else (
  set PYTHON_EXE=python
)

"%PYTHON_EXE%" "Source_code\main\diabetic_retinopathy\create_root_data_image_train_test_split.py" ^
  --root-data "ROOT_DATA" ^
  %*

set EXIT_CODE=%ERRORLEVEL%
endlocal & exit /b %EXIT_CODE%
