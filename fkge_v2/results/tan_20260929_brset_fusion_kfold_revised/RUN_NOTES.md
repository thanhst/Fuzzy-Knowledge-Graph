# FKG-E BRSET fusion, chạy lại 5-fold ngày 29/09/2026

## Giao thức

- Lệnh: `.\.venv\Scripts\python.exe -X utf8 -u fkge_v2\run_all.py --only kb1,baseline --brset-only --epochs 20 --run-id tan_20260929_brset_fusion_kfold_revised`.
- 5 fold theo patient ID từ gói FRB, 5 seed (42–46) mỗi fold, 20 epoch, không dùng `--quick`. Có 50 quan sát FKG-E (2 biến thể × 5 fold × 5 seed) trong `kb1_results.json`.
- Các phương pháp KB1 và baseline dùng cùng 5 fold validation của root train. Manifest `root_split` được đối chiếu lại theo số dòng, bệnh nhân, nhãn và giao bệnh nhân train/validation; overlap bằng 0 ở cả 5 fold.
- Train FRB đã qua SMOTE và không giữ patient ID thật cho từng dòng train. Root test 321 ảnh chưa có FRB tương ứng trong gói này. Đây là phân tích chẩn đoán 5-fold, chưa là đánh giá nested outer test.
- SGNS dùng đồng xuất hiện trên toàn luật; mỗi epoch lấy mẫu đều 2.000 cặp từ toàn bộ cặp trong luật. Suy diễn lấy điểm luật lớn nhất trong từng lớp rồi softmax. L2 có hệ số dương `1e-5` và được tính trong cả loss lẫn gradient.
- CI 95% là percentile bootstrap 10.000 lần theo fold sau khi lấy trung bình 5 seed trong fold. Fold là đơn vị lấy mẫu, không coi 25 lượt seed là độc lập.

## Kết quả trên cùng 5 fold

| Phương pháp | AUC-ROC TB ± SD (CI 95%) | BalAcc TB ± SD | F1 TB ± SD | Huấn luyện TB | Suy diễn ms/mẫu |
|---|---:|---:|---:|---:|---:|
| FISA tuần tự | 0,8231 ± 0,0791 (0,7568–0,8805) | 0,7141 ± 0,0718 | 0,2845 ± 0,0875 | 0,0064 s | 5,4913 |
| FISA bảng tra | 0,8231 ± 0,0791 (0,7568–0,8805) | 0,7141 ± 0,0718 | 0,2845 ± 0,0875 | 0,0059 s | 0,0485 |
| FKG-E không nhãn | 0,6055 ± 0,1904 (0,5674–0,6597) | 0,5458 ± 0,1024 | 0,1607 ± 0,0912 | 21,7081 s | 0,1653 |
| FKG-E có nhãn, objective hiện có | 0,9114 ± 0,0312 (0,8866–0,9370) | 0,8223 ± 0,0499 | 0,4761 ± 0,0898 | 21,5204 s trong KB1; 24,0986 s trong baseline | 0,1706 |
| MLP trên đặc trưng mờ | 0,9291 ± 0,0246 (0,9070–0,9488) | 0,8213 ± 0,0542 | 0,5761 ± 0,0459 | 0,4555 s | 0,0025 |

SD trong bảng là độ lệch chuẩn mẫu của 25 lượt fold × seed cho FKG-E/MLP, và của 5 fold cho FISA. CI không xem 25 lượt là độc lập.

Chênh AUC ghép cặp theo fold: FKG-E có nhãn trừ FISA = **+0,0883** (CI 0,0313–0,1452); MLP trừ FKG-E = **+0,0177** (CI −0,0111–0,0440). CI thứ hai chứa 0, nên chưa khẳng định MLP có AUC cao hơn một cách ổn định. Dữ liệu từng fold và cách lấy mẫu CI nằm trong `paired_comparisons.json`.

Độ trung thành của FKG-E có nhãn với FISA: đồng thuận **0,7159**, trung thành cân bằng **0,6927**, Cohen κ **0,3852**, KL trung bình **0,1309**. Điều kiện KL < γ²/2 bao phủ **0,0088** số truy vấn; không có vi phạm cận trong các truy vấn được bao phủ. FKG-E vẫn chưa đạt mục tiêu trung thành 0,9.

## Giới hạn và việc còn lại

Objective mới có `L_SGNS`, `L_node`, `L_inf`, `L_pred`, L2. Chưa có công thức nguồn để cài chính xác `L_edge`, `L_A`, `L_B`, `L_rule`; chưa hiệu chỉnh FISA bằng β/T hoặc chọn ngưỡng trên inner validation theo bệnh nhân. Chạy 20 epoch thay cho mặc định 100 để kiểm chứng giao thức 5 fold × 5 seed. Node2Vec chuẩn, XGBoost và PyKEEN chưa chạy; các bản `-lite` trong baseline không phải đối chứng chính thức. Không dùng số này làm kết luận cuối của luận án.

Tệp chi tiết: `run_manifest.json`, `kb1_results.json`, `baseline_comparison.json`, `paired_comparisons.json`, `report_tong_hop.md` và các `table_*.csv` trong cùng thư mục.

`source_changes_postrun.patch` lưu diff của working tree sau hai lượt chạy, gồm cả sửa câu chữ verdict/log và ghi chú được bổ sung sau run; không coi là snapshot chính xác tại giây bắt đầu KB1.
