@echo off
setlocal

set DATASET=%~1
if "%DATASET%"=="" set DATASET=both

set MODULE_DIR=%~2
if "%MODULE_DIR%"=="" set MODULE_DIR=source

set BINS=%~3
if "%BINS%"=="" set BINS=6

set WARM_REPEATS=%~4
if "%WARM_REPEATS%"=="" set WARM_REPEATS=1

echo [INFO] Running full FIS+FKG flow benchmark
echo [INFO] dataset=%DATASET%, module_dir=%MODULE_DIR%, bins=%BINS%, warm_repeats=%WARM_REPEATS%

python Source\tests\test_fis_fkg_full_flow_gpu_cpu.py ^
  --dataset %DATASET% ^
  --module-dir %MODULE_DIR% ^
  --bins %BINS% ^
  --warm-repeats %WARM_REPEATS% ^
  --out-json result\full_flow_benchmark_compact.json

if %errorlevel% neq 0 (
  echo [ERROR] Full-flow benchmark failed.
  exit /b %errorlevel%
)

echo [OK] Full-flow benchmark done.
echo [INFO] Report: result\full_flow_benchmark_compact.json
endlocal
