@echo off
setlocal
python run.py --models mlp --validate-data-only
if errorlevel 1 exit /b %errorlevel%
python run.py --models mlp --epochs 10 --batch-size 16 --results-dir outputs\mlp_5fold
