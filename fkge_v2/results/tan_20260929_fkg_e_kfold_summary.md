# Tổng hợp FKG-E BRSET fusion, 29/09/2026

## Đã chạy

Ba lượt BRSET fusion đều dùng 5 validation fold tách theo patient ID, 5 seed (42–46), 20 epoch, không dùng `--quick`. Bộ nạp kiểm tra lại nhãn, số mẫu và giao patient ID từ manifest; overlap train/validation bằng 0 ở cả 5 fold. Chưa đánh giá root test 321 ảnh.

| Phương pháp, cùng 5 fold | AUC-ROC TB ± SD (CI 95% theo fold) | BalAcc TB ± SD | F1 TB ± SD |
|---|---:|---:|---:|
| FISA bảng tra | 0,8231 ± 0,0791 (0,7568–0,8805) | 0,7141 ± 0,0718 | 0,2845 ± 0,0875 |
| FKG-E có nhãn, objective hiện có | 0,9114 ± 0,0312 (0,8866–0,9370) | 0,8223 ± 0,0499 | 0,4761 ± 0,0898 |
| MLP trên đặc trưng mờ | 0,9291 ± 0,0246 (0,9070–0,9488) | 0,8213 ± 0,0542 | 0,5761 ± 0,0459 |

SD là độ lệch chuẩn mẫu trên 25 lượt fold × seed cho FKG-E/MLP; FISA chỉ có 5 fold. CI bootstrap dùng 5 fold làm đơn vị lấy mẫu sau khi trung bình seed trong fold.

Chênh AUC ghép cặp: FKG-E trừ FISA **+0,0883** (CI 0,0313–0,1452); MLP trừ FKG-E **+0,0177** (CI −0,0111–0,0440). FKG-E có nhãn đạt đồng thuận với FISA **0,7159**, trung thành cân bằng **0,6927**, κ **0,3852**; chưa đạt mục tiêu 0,9.

KB2 với 30% luật và hạn ngạch tối thiểu 40% luật mỗi lớp: FKG-E AUC **0,9150**, BalAcc **0,8207** (so với 0,9114 và 0,8223 trên FKG đầy đủ). Nhưng FISA trên tập con rơi về BalAcc **0,5000** và đồng thuận FKG-E–FISA còn **0,1817**. Kết luận tự động cũ `cong_huong` đã được sửa thành `fisa_teacher_collapsed_fidelity_unresolved`. Một quét nhanh FISA-only cho thấy hạn ngạch cao hơn làm BalAcc giáo viên giảm; đây chưa phải cách lấy mẫu FKGS chính thức.

Quét λ_I/λ_P={0; 0,1; 1; 5}, giữ λ_P=20: AUC lần lượt **0,9109; 0,9114; 0,9140; 0,9185**, trung thành cân bằng **0,6920; 0,6927; 0,6976; 0,6997**, KL **0,1505; 0,1309; 0,0516; 0,0078**. KL giảm mạnh nhưng nhãn dự đoán chỉ tiến nhẹ; không điểm nào đạt cả AUC ≥ 0,90 và trung thành cân bằng ≥ 0,90. Đường này là chẩn đoán của objective hiện có.

## Mã đã chỉnh và giới hạn

SGNS dùng đồng xuất hiện toàn luật; dự đoán lấy điểm luật lớn nhất theo lớp; L2 được cộng vào loss; thêm trung thành cân bằng, κ, điều kiện biên KL và CI bootstrap theo fold; bộ nạp FRB kiểm tra patient ID/nhãn theo manifest; KB2 có hạn ngạch luật theo lớp; KB5 chuyển sang so toàn luật với cửa sổ `w=2`. Đã qua 10 kiểm thử, gradient check, `py_compile`, `git diff --check` và lượt kiểm tra KB5 hai chế độ.

Chưa có công thức nguồn để cài đúng `L_edge`, `L_A`, `L_B`, `L_rule`; FISA β/T và ngưỡng quyết định chưa hiệu chỉnh trên inner validation theo bệnh nhân vì FRB train sau SMOTE không giữ patient ID từng dòng. Chưa có FRB root test, baseline Node2Vec/XGBoost/PyKEEN chính thức, hoặc lượt chính thức 100 epoch. KB3–KB6 và ablation đầy đủ chưa chạy lại; không dùng bộ kết quả này để xác nhận đóng góp cấu trúc của Chương 3.

## Tệp kết quả

- [KB1 và baseline](tan_20260929_brset_fusion_kfold_revised/RUN_NOTES.md), [báo cáo tự sinh](tan_20260929_brset_fusion_kfold_revised/report_tong_hop.md), [chênh lệch ghép cặp](tan_20260929_brset_fusion_kfold_revised/paired_comparisons.json).
- [KB2 hạn ngạch](tan_20260929_brset_fusion_kfold_kb2_quota40/RUN_NOTES.md), [báo cáo](tan_20260929_brset_fusion_kfold_kb2_quota40/report_tong_hop.md).
- [Đường λ_I/λ_P](tan_20260929_brset_fusion_tradeoff_kfold/RUN_NOTES.md), [biểu đồ](tan_20260929_brset_fusion_tradeoff_kfold/tradeoff_auc_fidelity.png), [dữ liệu thô](tan_20260929_brset_fusion_tradeoff_kfold/tradeoff_results.json).
