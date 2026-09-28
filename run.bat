@echo off
setlocal
python run.py --validate-data-only
if errorlevel 1 exit /b %errorlevel%
python run.py --models all --resnet-arch resnet50 --epochs 10 --batch-size 16 --results-dir outputs\baseline_5fold
