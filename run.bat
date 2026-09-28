@echo off
setlocal
cd /d "%~dp0"
python run.py --check-reference
exit /b %errorlevel%
