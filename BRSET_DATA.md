# Dữ liệu BRSET đầy đủ dùng cho các lần chạy mới

Dùng [reproducibility/brset_full_features/data](reproducibility/brset_full_features/data):
**16.266 ảnh duy nhất / 8.524 bệnh nhân**, gồm 15.196 ảnh không DR và 1.070 ảnh có DR.
Đây là đặc trưng đã lưu từ ảnh và bảng lâm sàng; thư mục không chứa ảnh gốc.

- [fusion_features.csv](reproducibility/brset_full_features/data/fusion_features.csv): 23 đặc trưng, có `image_id`, `patient_id` và nhãn từng ảnh.
- [image_features.csv](reproducibility/brset_full_features/data/image_features.csv): 10 đặc trưng ảnh.
- [table_features.csv](reproducibility/brset_full_features/data/table_features.csv): 13 biến lâm sàng, giữ giá trị thiếu.
- [fold_assignments.csv](reproducibility/brset_full_features/data/fold_assignments.csv): toàn bộ ảnh được chia thành 5 fold bằng `StratifiedGroupKFold`, seed 42, nhóm theo `patient_id`.
- `data/fold_01` đến `data/fold_05`: mỗi thư mục chứa `train_ids.csv` và `test_ids.csv`. Ghép với bảng đặc trưng qua `image_id`.

Mỗi lượt có khoảng 13.000 ảnh của 6.820 bệnh nhân để huấn luyện và 3.250 ảnh
của 1.705 bệnh nhân để kiểm tra. Không có bệnh nhân giao nhau giữa train/test;
mỗi ảnh nằm trong test đúng một lần sau năm lượt.

Xem [hướng dẫn, bảng đếm từng fold, môi trường và lệnh chạy](reproducibility/brset_full_features/README.md).
Gói này có công cụ kiểm tra và tạo lại fold độc lập với code huấn luyện của từng nhánh:

```bash
python -m pip install -r reproducibility/brset_full_features/requirements-lock.txt
python -B -X utf8 reproducibility/brset_full_features/build.py --verify-only
python -B -X utf8 reproducibility/brset_full_features/build.py --output reproducibility/brset_full_features/private/rebuild --folds 5 --seed 42
git rev-parse HEAD
```

Các file kết quả và các ví dụ chạy cũ trong từng nhánh vẫn mô tả lần chạy cũ,
có thể chỉ dùng khoảng 1.500 ảnh. Muốn chạy toàn bộ BRSET phải nạp CSV và fold
trong gói này; việc cập nhật dữ liệu chưa tạo ra kết quả huấn luyện FKG-E mới.
Không đưa cột ID, nhãn hoặc fold vào đầu vào mô hình. Tạo inner validation theo
bệnh nhân trong từng outer train; chỉ fit tiền xử lý và SMOTE trên inner train,
chọn alpha và các siêu tham số bằng validation, rồi đánh giá outer test.
