@echo off
setlocal
echo Script nay da duoc thay bang pipeline moi.
echo Dang goi Build_FISA_CPU.bat ...
call "%~dp0Build_FISA_CPU.bat"
exit /b %errorlevel%
