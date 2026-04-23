#!/bin/bash
set -e  # dừng ngay khi có lệnh lỗi

# --- Logtime
echo "[INFO] --- Script started at: $(date)"

# --- set workdir
cd Source_code || exit 1

# --- install code FKG
pip install --find-links=module/Setup_module/CMAKE/wheel/wheelhouse/linux fisa_module

# --- Running the Python script...
echo "[INFO] --- Running the Python script..."

python3 -m main.Duy_Hoang_progress.fusion_wrapper
if [ $? -ne 0 ]; then
    echo "[ERROR] --- Python script execution failed. Check the error above."
    exit 1
fi

# --- finish
echo "[INFO] --- Python script executed successfully."
echo "[INFO] --- Script finished at: $(date)"
