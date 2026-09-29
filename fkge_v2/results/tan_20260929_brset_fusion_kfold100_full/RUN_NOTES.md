# FKG-E BRSET fusion 100 epoch: KB1 và KB2

Lượt chạy 5 fold theo patient ID, 5 seed (42–46), 100 epoch, `quick=false`. Dữ liệu là FRB BRSET fusion ảnh và bảng. Bộ nạp đối chiếu manifest nguồn; giao patient ID train/validation bằng 0 ở mọi fold. `kb1_results.json` có 50 quan sát FKG-E (hai biến thể × 5 fold × 5 seed).

| Phương pháp / tập luật | AUC-ROC TB ± SD (CI 95%) | BalAcc TB ± SD | F1 TB ± SD |
|---|---:|---:|---:|
| FISA bảng tra / FKG đầy đủ | 0,8231 ± 0,0791 (0,7568–0,8805) | 0,7141 ± 0,0718 | 0,2845 ± 0,0875 |
| FKG-E có nhãn / FKG đầy đủ | 0,9305 ± 0,0170 (0,9149–0,9437) | 0,8411 ± 0,0408 | 0,5142 ± 0,0560 |
| FISA bảng tra / 30% luật | 0,7693 ± 0,0992 (0,6943–0,8455) | 0,5000 ± 0,0000 | 0,1487 ± 0,0039 |
| FKG-E có nhãn / 30% luật | 0,9333 ± 0,0184 (0,9166–0,9482) | 0,8439 ± 0,0503 | 0,5114 ± 0,0447 |

SD là độ lệch chuẩn mẫu của 25 lượt fold × seed đối với FKG-E, và 5 fold đối với FISA. CI 95% là bootstrap 10.000 lần theo fold, sau khi trung bình seed trong từng fold.

Độ trung thành trên FKG đầy đủ: đồng thuận 0,7033, trung thành cân bằng 0,6685, κ 0,3498. Khi rút xuống 30% luật với hạn ngạch tối thiểu 40% mỗi lớp, FISA dự đoán một lớp (BalAcc 0,5) và đồng thuận FKG-E–FISA còn 0,1751; đây không xác nhận H-E2 về độ trung thành. Tập 30% là lấy mẫu mô phỏng, chưa phải FKGS chính thức.

Nguồn mã lúc chạy: commit `db2d451` (`source_dirty=false`). Tiến trình ban đầu cũng được yêu cầu chạy KB3–KB6, ablation và baseline, nhưng đã dừng sau KB2 để bổ sung quan sát từng fold/seed và CI cho các kịch bản này. Hai kịch bản KB1/KB2 hoàn tất và được lưu nguyên; các kịch bản còn lại chạy ở run ID riêng từ commit `0c06757`.

Mô hình vẫn thiếu `L_edge`, `L_A`, `L_B`, `L_rule`; chưa hiệu chỉnh FISA β/T hay ngưỡng trên inner validation theo bệnh nhân. Gói FRB chưa có root test tương ứng. Đây là đánh giá trên validation của root train, chưa phải kết luận chính thức của Chương 3.

Chi tiết: `run_manifest.json`, `kb1_results.json`, `kb2_results.json`, `report_tong_hop.md`, `table_KB1.csv`, `table_KB2.csv`, `figures/`.
