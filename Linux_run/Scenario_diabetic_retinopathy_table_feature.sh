#!/bin/bash

# Ghi lại thời gian bắt đầu để dễ dàng kiểm tra
echo "[INFO] --- Script started at: $(date)"

# Đặt thư mục làm việc
cd ..
cd Source_code

# Cài đặt module fisa_module từ wheelhouse
pip install --find-links=module/Setup_module/wheelhouse fisa_module

# Hiển thị thông báo trước khi chạy script
echo "[INFO] --- Running the Python script..."

# Chạy Python script với module
python -m main.diabetic_retinopathy.Scenario_diabetic_retinopathy_table_feature

# Kiểm tra nếu lệnh python chạy thành công
if [ $? -ne 0 ]; then
    echo "[ERROR] --- Python script execution failed. Check the error above."
    echo "[INFO] --- Press any key to exit..."
    read -n 1 -s -r
    exit 1
fi

# Hiển thị thông báo khi chạy thành công
echo "[INFO] --- Python script executed successfully."

# Giữ cửa sổ terminal mở để kiểm tra kết quả
echo "[INFO] --- Press any key to exit..."
read -n 1 -s -r

# Ghi lại thời gian kết thúc
echo "[INFO] --- Script finished at: $(date)"
