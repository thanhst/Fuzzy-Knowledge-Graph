# Đặc trưng đầy đủ BRSET và 5 fold theo bệnh nhân

Đã tìm thấy dữ liệu đặc trưng của **16.266 ảnh / 8.524 bệnh nhân** trong Git.
Nguồn đặc trưng ảnh là [Fusion_feature/images_ft.csv tại bản nguồn](https://github.com/thanhst/Fuzzy-Knowledge-Graph/blob/5961fdfe40ae85df50cae8c15ef3ae960c2f1edd/Source_code/data/Dataset_diabetic/Fusion_feature/images_ft.csv):
16.266 mã ảnh duy nhất, mỗi ảnh có 10 đặc trưng số hữu hạn, khớp đủ bảng nhãn.
File này được cập nhật lần cuối tại commit
[`9786ba5`](https://github.com/thanhst/Fuzzy-Knowledge-Graph/commit/9786ba5162919fa42c3cc0f34db52092163040fb)
Gói CSV đầy đủ trong thư mục này được đồng bộ lên tất cả các nhánh hiện có
trên `origin`. Có thể dùng gói này độc lập với cấu trúc code huấn luyện của từng nhánh.

Đợt kiểm tra đọc 1.837 phiên bản CSV trong 168 commit; 13 phiên bản CSV có
đủ 16.266 mã ảnh BRSET duy nhất. Xem [history_audit.json](history_audit.json).
Việc thiếu ảnh gốc trong thư mục hiện tại không đồng nghĩa thiếu các đặc trưng
đã trích trước đây: có thể dùng cache này để chạy mô hình trên đặc trưng mà
không tải lại toàn bộ ảnh. Các đặc trưng chưa được đối chiếu lại bằng cách
trích lại từ đủ ảnh gốc trong đợt kiểm tra này.

## Tải CSV nào?

| File | Dòng | Đặc trưng | Nội dung |
|---|---:|---:|---|
| [image_features.csv](data/image_features.csv) | 16.266 | 10 | Đặc trưng ảnh chưa chuẩn hóa toàn cục |
| [table_features.csv](data/table_features.csv) | 16.266 | 13 | Biến lâm sàng từ bảng nhãn gốc, còn giá trị thiếu |
| [fusion_features.csv](data/fusion_features.csv) | 16.266 | 23 | Ghép ảnh và lâm sàng bằng `image_id` |
| [fold_assignments.csv](data/fold_assignments.csv) | 16.266 | — | Mã ảnh, mã bệnh nhân, nhãn và fold giữ lại |

Ba bảng đặc trưng có `image_id`, `patient_id`, `diabetic_retinopathy` bên cạnh
các cột đầu vào. Nhãn được giữ theo từng ảnh: **0 = không DR, 1 = có DR**.
Số ảnh theo lớp là 15.196 âm và 1.070 dương. Các cột ID/nhãn không phải đặc trưng.
Số fold chỉ nằm trong file phân chia, tránh đưa nhầm vào mô hình.

Mỗi thư mục `data/fold_01` đến `data/fold_05` có `train_ids.csv` và
`test_ids.csv`; dùng `image_id` để lấy các dòng tương ứng từ bảng đặc trưng.
Đây là **dữ liệu đầu vào và bản chia fold**, chưa phải luật mờ hoặc kết quả
huấn luyện FKG-E trên toàn bộ BRSET. Kết quả FKG-E 1.208 ảnh đã công bố trước
đó vẫn thuộc phạm vi dữ liệu riêng của lần chạy đó.

## Kết quả chia fold

Dùng `StratifiedGroupKFold` của scikit-learn với
`n_splits=5`, `shuffle=True`, `random_state=42`, `groups=patient_id` và
nhãn DR của từng ảnh để phân tầng. Không fallback sang `GroupKFold`.
Helper chạy độc lập được đối chiếu với bản phát hành: tạo lại đúng từng byte
của cả 14 file CSV, bao gồm các fold đã công bố.

| Fold | Ảnh train | Ảnh test | Bệnh nhân train | Bệnh nhân test | Ảnh DR dương ở test |
|---|---:|---:|---:|---:|---:|
| 1 | 13.006 | 3.260 | 6.819 | 1.705 | 198 |
| 2 | 13.014 | 3.252 | 6.819 | 1.705 | 219 |
| 3 | 13.018 | 3.248 | 6.818 | 1.706 | 239 |
| 4 | 13.011 | 3.255 | 6.821 | 1.703 | 204 |
| 5 | 13.015 | 3.251 | 6.819 | 1.705 | 210 |

Đã xác nhận bệnh nhân train/test không giao nhau trong cả năm lượt;
mỗi ảnh xuất hiện đúng một lần trong các tập test, mỗi bệnh nhân chỉ thuộc
một fold giữ lại. Toàn bộ 8.524 bệnh nhân tham gia quy trình 5 fold này;
không tách thêm một tập root test trước khi chia.

## Vì sao không lấy thẳng mọi file tiền xử lý cũ?

File `Fusion_feature/data_process.csv` cũng có 16.266 dòng nhưng không giữ ID,
đã điền giá trị thiếu/chuẩn hóa trên toàn bộ dữ liệu và mã hóa nhãn DR thành
1/2. Nhiều file 30.392 dòng là dữ liệu đã cân bằng, không phải 30.392 ảnh gốc.
Các file PCA, chọn thuộc tính hoặc `image_fts_norm.csv` cũng đã có bước học
tham số trên dữ liệu trước khi chia lại.

Script xuất lần này chỉ dùng các đặc trưng ảnh từng ảnh được lưu trước bước
chuẩn hóa toàn cục, rồi ghép `patient_id` và nhãn gốc bằng `image_id`.
Biến lâm sàng lấy từ `ROOT_DATA/labels_brset.csv`, không lấy bản đã điền thiếu.
Chỉ áp dụng ánh xạ cố định như `yes/no → 2/1`,
`Adequate/Inadequate → 2/1` và ép số; giá trị thiếu/không hợp lệ vẫn để trống.
Ví dụ tuổi thiếu 5.446 dòng, thời gian mắc bệnh thiếu 14.402 dòng;
không loại các ảnh đó hoặc điền bằng trung bình toàn bộ dữ liệu.

Khi chạy mô hình, cần chia inner train/validation theo bệnh nhân trước khi fit
điền thiếu, scaler, lựa chọn thuộc tính, học hợp nhất, mờ hóa, SMOTE, luật và
FKG. SMOTE chỉ dùng inner train. Chọn alpha, các siêu tham số, ngưỡng và điểm
dừng trên validation; outer test chỉ dùng để đánh giá lựa chọn đã khóa.
Các CSV trong thư mục này chưa thực hiện các bước học đó.

## Dùng trên Colab

Sau khi clone repository và chọn nhánh cần dùng,
đọc CSV trực tiếp; không cần tải ảnh để sử dụng các đặc trưng đã lưu:

```python
import pandas as pd

base = "reproducibility/brset_full_features/data"
features = pd.read_csv(f"{base}/fusion_features.csv",
                       dtype={"image_id": str, "patient_id": str})
assignments = pd.read_csv(f"{base}/fold_assignments.csv",
                          dtype={"image_id": str, "patient_id": str})
data = features.merge(assignments[["image_id", "fold"]],
                      on="image_id", validate="one_to_one")

k = 1
train = data.loc[data["fold"] != k].copy()
test = data.loc[data["fold"] == k].copy()
assert not set(train.patient_id) & set(test.patient_id)
feature_columns = [c for c in data.columns
                   if c not in {"image_id", "patient_id", "diabetic_retinopathy", "fold"}]
X_train, y_train = train[feature_columns], train.diabetic_retinopathy
X_test, y_test = test[feature_columns], test.diabetic_retinopathy
# Tạo inner validation theo patient_id trước mọi bước fit tiếp theo.
```

## Tái lập việc xuất CSV và kiểm tra

Chạy từ thư mục gốc repository, dùng Python 3.13.7 và
[requirements-lock.txt](requirements-lock.txt).
Output phải mới/rỗng; script không ghi đè dữ liệu có sẵn.
Các lệnh này dùng CSV đi kèm, không cần thư mục ảnh gốc, bảng nhãn bên ngoài
hoặc helper từ nhánh khác. Việc chia fold chỉ dùng CPU.

```bash
python -m pip install -r reproducibility/brset_full_features/requirements-lock.txt
python -B -X utf8 reproducibility/brset_full_features/build.py --verify-only
python -B -X utf8 reproducibility/brset_full_features/build.py --output reproducibility/brset_full_features/private/rebuild --folds 5 --seed 42
python -B -X utf8 -m unittest discover -s reproducibility/brset_full_features/tests -v
```

[release_manifest.json](data/release_manifest.json) ghi seed, môi trường,
hash dữ liệu và code, Git blob/commit nguồn, số thiếu, các kiểm tra phân chia
và SHA256 của 14 file CSV. Giữ nguyên manifest của đợt xuất dữ liệu gốc;
hash code trong manifest đó tương ứng với [builder tại commit 5961fdf](https://github.com/thanhst/Fuzzy-Knowledge-Graph/blob/5961fdfe40ae85df50cae8c15ef3ae960c2f1edd/reproducibility/brset_full_features/build.py).
Builder hiện tại dùng `portable_folds.py` trong cùng thư mục, gọi trực tiếp
scikit-learn. Mỗi lần tạo lại sẽ ghi manifest mới với hash code đang chạy,
hash CSV đầu vào và phiên bản môi trường; 14 CSV giữ nguyên với seed 42
và môi trường đã khóa. Lưu thêm `git rev-parse HEAD` khi chạy thí nghiệm.

Nếu cần ghép lại từ hai file nguồn, tại checkout có đủ nguồn gốc:

```bash
python -B -X utf8 reproducibility/brset_full_features/build.py --image-source Source_code/data/Dataset_diabetic/Fusion_feature/images_ft.csv --labels ROOT_DATA/labels_brset.csv --output reproducibility/brset_full_features/private/from_sources --folds 5 --seed 42
```
