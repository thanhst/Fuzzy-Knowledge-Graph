# KB2 BRSET fusion, lấy 30% luật có hạn ngạch lớp

- Lệnh: `.\.venv\Scripts\python.exe -X utf8 -u fkge_v2\run_all.py --only kb2 --brset-only --epochs 20 --run-id tan_20260929_brset_fusion_kfold_kb2_quota40`.
- 5 fold theo patient ID, 5 seed, 20 epoch. FKG đầy đủ có trung bình 1.054,6 luật; tập con có 316,6 luật. Tập con dành tối thiểu 40% số luật cho **mỗi** lớp. Số luật lớp dương qua 5 fold: 144, 157, 133, 152, 134. Đây là mẫu ngẫu nhiên có hạn ngạch, không phải S-FKGS chính thức.

| Chỉ số (TB ± SD) | FKG đầy đủ | 30% luật có hạn ngạch |
|---|---:|---:|
| FKG-E có nhãn AUC-ROC | 0,9114 ± 0,0312 | 0,9150 ± 0,0308 |
| FKG-E có nhãn BalAcc | 0,8223 ± 0,0499 | 0,8207 ± 0,0610 |
| FKG-E có nhãn F1 | 0,4761 ± 0,0898 | 0,4830 ± 0,0929 |
| FKG-E–FISA đồng thuận | 0,7159 ± 0,1371 | 0,1817 ± 0,0503 |
| FISA bảng tra AUC-ROC | 0,8231 ± 0,0791 | 0,7693 ± 0,0992 |
| FISA bảng tra BalAcc | 0,7141 ± 0,0718 | 0,5000 ± 0,0000 |
| FKG-E ms/mẫu | 0,1676 | 0,1420 |

SD là độ lệch chuẩn mẫu trên 25 lượt fold × seed cho FKG-E và trên 5 fold cho FISA. SD bằng 0 ở FISA rút gọn vì BalAcc bằng 0,5 trong cả 5 fold; đây là dấu hiệu mô hình sụp về một lớp.

Chênh lệch ghép cặp của tập con trừ FKG đầy đủ, bootstrap theo 5 fold sau khi lấy trung bình seed: AUC **+0,0036** (CI 0,0007–0,0064), BalAcc **−0,0016** (CI −0,0174–0,0147), đồng thuận **−0,5342** (CI −0,6561 đến −0,4168). Tốc độ suy diễn FKG-E nhanh hơn khoảng 1,18 lần.

Hạn ngạch cao làm FISA trên tập rút gọn dự đoán gần một lớp (BalAcc 0,5), nên sự giữ được AUC/BalAcc của FKG-E **không** xác nhận đóng góp trung thành với FISA hay giả thuyết H-E2 đầy đủ. Trường `legacy_interpretation=cong_huong` trong JSON là kết luận tự động cũ chỉ xét AUC và tốc độ; trường `interpretation=fisa_teacher_collapsed_fidelity_unresolved` là đánh giá đã sửa dựa trên BalAcc giáo viên. Verdict được tính lại từ các hàng kết quả sau khi runner hoàn tất.

Kiểm tra nhanh FISA với cùng 30% luật và các mức hạn ngạch `{0, 0,2, 0,3, 0,35, 0,4}` cho BalAcc trung bình lần lượt `{0,6017; 0,5390; 0,5000; 0,5022; 0,5000}`. Điều này cho thấy tăng hạn ngạch dương trong cách lấy mẫu này gây lệch FISA; cần hiệu chỉnh giáo viên và đánh giá nested trước khi chọn hạn ngạch.

Tệp chi tiết: `run_manifest.json`, `kb2_results.json`, `report_tong_hop.md`, `table_KB2.csv` trong cùng thư mục.

`source_changes_postrun.patch` là diff của working tree sau run; bao gồm sửa verdict/log và tài liệu sau chạy nên không phải snapshot chính xác tại lúc bắt đầu.
