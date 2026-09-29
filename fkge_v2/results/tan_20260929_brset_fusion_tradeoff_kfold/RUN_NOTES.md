# Đường λ_I/λ_P của FKG-E trên BRSET fusion, 5-fold

- Lệnh: `.\.venv\Scripts\python.exe -X utf8 -u fkge_v2\run_fidelity_tradeoff.py --kb1-run-id tan_20260929_brset_fusion_kfold_revised --run-id tan_20260929_brset_fusion_tradeoff_kfold --epochs 20`.
- 5 fold theo bệnh nhân × 5 seed (42–46), 20 epoch, λ_P=20 cố định, λ_I={0,2,20,100}; tổng 100 quan sát. Điểm λ_I=2 tái dùng 25 quan sát KB1 có manifest khớp. CI 95% dùng bootstrap 10.000 lần theo fold sau khi lấy trung bình seed trong fold.
- SGNS đồng xuất hiện toàn luật, cực đại theo lớp và L2 dương như lượt KB1. Giáo viên FISA và ngưỡng quyết định chưa hiệu chỉnh trên inner validation; 4 loss `L_edge`, `L_A`, `L_B`, `L_rule` chưa có công thức/cài đặt. Đây là đường chẩn đoán của objective hiện có, không là kết quả cuối của luận án.

| λ_I/λ_P | AUC-ROC TB ± SD (CI 95%) | BalAcc | Đồng thuận | Trung thành cân bằng TB ± SD (CI 95%) | κ | KL TB |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,9109 ± 0,0314 (0,8861–0,9366) | 0,8189 | 0,7152 | 0,6920 ± 0,0880 (0,6190–0,7687) | 0,3842 | 0,1505 |
| 0,1 | 0,9114 ± 0,0312 (0,8866–0,9370) | 0,8223 | 0,7159 | 0,6927 ± 0,0878 (0,6198–0,7691) | 0,3852 | 0,1309 |
| 1 | 0,9140 ± 0,0302 (0,8900–0,9390) | 0,8380 | 0,7205 | 0,6976 ± 0,0867 (0,6247–0,7713) | 0,3935 | 0,0516 |
| 5 | 0,9185 ± 0,0281 (0,8962–0,9420) | 0,8436 | 0,7232 | 0,6997 ± 0,0859 (0,6279–0,7723) | 0,3972 | 0,0078 |

SD là độ lệch chuẩn mẫu của 25 lượt fold × seed mỗi mức λ_I; CI vẫn bootstrap theo 5 fold sau khi trung bình seed. `table_tradeoff.csv` có SD cho mọi chỉ số. Các SD được bổ sung sau khi chạy từ 100 quan sát gốc; không huấn luyện lại mô hình.

So sánh ghép cặp tỉ lệ 5 trừ 0, bootstrap theo 5 fold: AUC **+0,0075** (CI 0,0044–0,0115); BalAcc **+0,0247** (CI 0,0067–0,0427); đồng thuận **+0,0079** (CI −0,0012–0,0187); trung thành cân bằng **+0,0077** (CI 0,0021–0,0143); KL **−0,1428** (CI −0,1667 đến −0,1242).

KL giảm mạnh nhưng nhãn dự đoán của FKG-E ít đổi vì biên của FISA nhỏ. Trong bốn mức đã thử, không có vùng AUC ≥ 0,90 và trung thành cân bằng ≥ 0,90. Không diễn giải đường này như bằng chứng của một điểm cân bằng tốt trong mô hình Chương 3 đầy đủ.

Tệp chi tiết: `run_manifest.json`, `tradeoff_results.json`, `table_tradeoff.csv`, `tradeoff_auc_fidelity.png`; mã chạy được sao vào `run_fidelity_tradeoff_source.py`. Root test 321 ảnh vẫn chưa có FRB để đánh giá ngoài cùng.
