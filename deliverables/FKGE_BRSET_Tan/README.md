# Gói riêng cho thầy Hồng Tân chạy FKG-E trên BRSET đầy đủ

Thư mục này chạy độc lập: **16.266 ảnh / 8.524 bệnh nhân**, đủ code, CSV,
5 fold và lệnh xuất bảng/ảnh. Không cần tải ảnh gốc hoặc tìm file trong nhánh khác.
Đầu vào là đặc trưng đã trích trước đây; đây chưa phải kết quả huấn luyện FKG-E mới.
Bảng [CHECKED_FOLD_COUNTS.png](CHECKED_FOLD_COUNTS.png) và CSV cùng tên đã được
xuất từ bước chuẩn bị thật của đủ 5 fold, có phân biệt rõ outer/inner train.

## Chạy trên Colab

Mở [notebook Colab](https://colab.research.google.com/github/thanhst/Fuzzy-Knowledge-Graph/blob/main/deliverables/FKGE_BRSET_Tan/CHAY_TREN_COLAB.ipynb),
chọn GPU trong Runtime rồi chạy lần lượt các ô. Notebook tải đúng gói ZIP riêng,
cài thư viện, kiểm tra dữ liệu và chạy cả 5 fold. Kết quả/checkpoint được lưu
trên Google Drive của người chạy; cache teacher lớn nằm trong ổ tạm của Colab.

Lưới đầy đủ có **108 cấu hình/fold**, tối đa 200 epoch/cấu hình, early stopping
patience 20. Chưa đo thời gian chạy đầy đủ trên Colab; lượt này chỉ kiểm thử CPU
với dữ liệu tổng hợp và kiểm tra tiền xử lý của 5 fold dữ liệu thật.
Notebook có `--resume`: chạy lại cùng lệnh sẽ tiếp tục từ cấu hình đã hoàn tất.
Thay code, dữ liệu, seed, môi trường hoặc cấu hình thì cần output mới.

## Chạy trên máy riêng

Mở terminal ngay trong thư mục `FKGE_BRSET_Tan`. Môi trường kiểm thử:
Python 3.13.7, PyTorch 2.11.0+cu130 (chạy CPU), các phiên bản số học trong
`requirements-common.txt`. `requirements.txt` khóa phiên bản PyTorch;
`requirements-colab.txt` dùng PyTorch/CUDA có sẵn của Colab và ghi phiên bản thực tế vào manifest.

```bash
python -m pip install -r requirements.txt
python -B -X utf8 run.py --check
python -B -X utf8 -u run.py --device auto --output outputs/brset_full --resume
```

Có thể dùng môi trường ảo riêng. Trên máy chỉ chạy CPU, cài wheel CPU trước:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Chạy thử nhỏ để xác nhận môi trường, **không tạo kết quả BRSET**:

```bash
python -B -X utf8 -m unittest discover -s tests -v
python -B -X utf8 -u run.py --smoke --folds 1 --device cpu --output outputs/smoke --resume
```

Chỉ chuẩn bị dữ liệu và xuất bảng đếm của đủ 5 fold:

```bash
python -B -X utf8 -u run.py --stage prepare --device auto --output outputs/brset_full --resume
```

Sau đó chạy lệnh đầy đủ bên trên, giữ nguyên `--device` và output để tiếp tục.
Nếu RAM GPU hạn chế, dùng `--block-size 64` ngay từ đầu; block chỉ chia phép tính,
không thay gradient toàn bộ hay số mẫu thống kê. Chạy CPU được nhưng chưa có số đo
thời gian đầy đủ. Teacher dùng file memory map trong thư mục tạm, có thể cần vài GiB
ổ đĩa; mất cache tạm thì gói tự tính lại từ dữ liệu chuẩn bị đã lưu.

## Đầu ra gửi lại

Trong `outputs/brset_full/reports/` hoặc thư mục Drive tương ứng:

- `fold_population.csv` và `.png`: ảnh, bệnh nhân ở outer train/inner train/validation/test,
  số luật gồm SMOTE, số mẫu SMOTE, số nhóm teacher và nhóm lớn nhất.
- `metrics_by_fold.csv` và `.png`: FISA và FKG-E trên outer test từng fold.
- `metrics_mean_std.csv` và `.png`: trung bình và độ lệch chuẩn mẫu giữa các outer fold.
- `metrics_pooled_oof.csv`, `predictions_oof.csv`, `roc_pr.png`: chỉ số và đường ROC/PR
  từ các dự đoán ngoài fold; mỗi ảnh được đánh giá đúng một lần sau đủ 5 fold.
- `selected_hyperparameters.csv`: alpha, chiều embedding, lambda I/P, validation loss và epoch đã chọn.
- `confusion_fisa.csv`, `confusion_fkge.csv`: ma trận nhầm lẫn.

`average_precision` là AP của scikit-learn; `pr_auc_trapezoidal` là diện tích PR
tính theo hình thang. Báo riêng cả hai, không gọi hai cách tính là cùng một chỉ số.
Mean/std theo fold khác chỉ số pooled OOF. Khi mới chạy một phần fold, báo cáo
ghi đúng số fold hoàn tất; cần đủ 5 fold để báo cáo thí nghiệm này.

Mỗi `fold_XX/` giữ danh sách ID thật, nguồn hai mẫu cha của SMOTE, tiền xử lý,
checkpoint từng cấu hình, lịch sử học, cấu hình được chọn và dự đoán test.
`run_manifest.json` ghi code/commit hoặc thông tin phát hành ZIP, seed, SHA256,
phiên bản thư viện, CPU/GPU/RAM, câu lệnh chính xác và lần resume.
`environment-freeze.txt` ghi môi trường thực tế để tái lập.

## Giao thức đã nối trong gói

1. Outer split giữ nguyên `StratifiedGroupKFold(5, shuffle=True, random_state=42)`
   theo `patient_id` trong CSV đã xác minh; không lấy lại bộ khoảng 1.500 ảnh.
2. Trong mỗi outer train, lấy một phần của SGKF 5 phần làm validation thật,
   seed `42 + fold`; không trùng bệnh nhân giữa inner train, validation và test.
3. Fit median imputer, StandardScaler, chọn 7 đặc trưng ảnh + 9 lâm sàng bằng ANOVA
   chỉ trên inner train. Số đặc trưng, FCM 5 mức và các hằng số trong backend là
   lựa chọn cố định từ trước thí nghiệm, không chọn bằng test.
4. Chia inner train thành 5 phần teacher theo bệnh nhân, rồi chạy BorderlineSMOTE
   riêng trong từng phần. Mẫu tổng hợp và cả hai mẫu cha ở cùng phần; giá trị thiếu,
   scaler và lựa chọn thuộc tính không fit lại trên validation/test. FCM fit trên
   dữ liệu huấn luyện sau SMOTE; teacher của một phần chỉ dùng luật từ bốn phần khác.
5. Giữ nhóm theo bệnh nhân và nguồn SMOTE. Hai bệnh nhân thật có cùng luật mờ
   vẫn là hai bệnh nhân; chỉ gộp luật giống nhau trong cùng nhóm, giữ trọng số số lần xuất hiện.
6. Chọn alpha, d_e, lambda I/P, epoch, ngưỡng và nhiệt độ hiệu chỉnh bằng validation.
   Không dùng outer-test loss/AUC/AP để chọn. Khóa lựa chọn trước khi đánh giá test;
   không refit gộp validation trong giao thức này.
7. Đơn vị đánh giá là 5 outer fold. Các phần teacher, nhóm nối cha SMOTE và block
   tính toán **không được coi là các mẫu độc lập** để tăng n hoặc tính CI.

Backend dùng float64 và gradient toàn bộ có chia block để giảm RAM. Kiểm tra
trong `tests/test_pipeline.py` đối chiếu giá trị FISA, attribution, loss và gradient
FKG-E với code gốc trong `reference/`, gồm trường hợp luật trùng có trọng số.
Đây là cùng hàm mục tiêu trên cùng cơ sở luật; kết quả thí nghiệm cũ không được kỳ vọng
khớp vì bộ dữ liệu và giao thức validation/SMOTE đã thay đổi.

`SOURCE_PROVENANCE.json` ghi hash zip nguồn và hai file mô hình gốc giữ nguyên.
`data/release_manifest.json` giữ bằng chứng của đợt xuất CSV ban đầu;
hash code trong manifest dữ liệu là code xuất CSV của lần đó.
`build.py` và `portable_folds.py` dùng kiểm tra/tạo lại CSV, không dùng để huấn luyện.
Gói chỉ gồm thí nghiệm FKG-E/FISA trên BRSET, không chạy KB2–KB6 hay baseline khác.
