@echo off
setlocal enabledelayedexpansion

rem --- Ghi lại thời gian bắt đầu để dễ dàng kiểm tra
echo [INFO] --- Script started at: %date% %time%

rem --- Đặt thư mục làm việc
cd ../Source_code

rem --- install code FKG
pip install --find-links=module/Setup_module/CMAKE/wheel/wheelhouse/window fisa_module


rem --- Hiển thị thông báo trước khi chạy script
echo [INFO] --- Running the Python script...

rem --- Chạy Python script với module
python -m test.diabetic_retinopathy.Scenario_diabetic_retinopathy_fusion_feature

rem --- Kiểm tra nếu lệnh python chạy thành công
if %errorlevel% neq 0 (
    echo [ERROR] --- Python script execution failed. Check the error above.
    echo [INFO] --- Press any key to exit...
    pause
    exit /b
)

rem --- Hiển thị thông báo khi chạy thành công
echo [INFO] --- Python script executed successfully.

rem --- Giữ cửa sổ terminal mở để kiểm tra kết quả
echo [INFO] --- Press any key to exit...
pause

rem --- Ghi lại thời gian kết thúc
echo [INFO] --- Script finished at: %date% %time%
