# FKG-E BRSET fusion 100 epoch: đường đánh đổi độ chính xác – độ trung thành

Chạy trên 5 fold tách theo patient ID × 5 seed (42–46), 100 epoch, `quick=false`. Mỗi trong 4 cấu hình có 25 quan sát fold × seed; cấu hình mặc định λ_I=2 được tái sử dụng từ KB1 cùng giao thức. SD là độ lệch chuẩn mẫu trên 25 quan sát; CI 95% là bootstrap 10.000 lần theo fold sau khi lấy trung bình seed trong fold.

| λ_I/λ_P | AUC-ROC TB ± SD (CI 95%) | Trung thành cân bằng TB ± SD (CI 95%) | Đồng thuận TB | KL TB |
|---:|---:|---:|---:|---:|
| 0 | 0,9300 ± 0,0155 (0,9158–0,9416) | 0,6633 ± 0,0665 (0,6119–0,7215) | 0,6999 | 1,5092 |
| 0,1 | 0,9305 ± 0,0170 (0,9149–0,9437) | 0,6685 ± 0,0686 (0,6144–0,7280) | 0,7033 | 0,7206 |
| 1 | 0,9311 ± 0,0203 (0,9132–0,9482) | 0,6791 ± 0,0745 (0,6192–0,7407) | 0,7098 | 0,1020 |
| 5 | 0,9310 ± 0,0230 (0,9106–0,9507) | 0,6878 ± 0,0725 (0,6278–0,7478) | 0,7194 | 0,0097 |

Độ trung thành tăng ít dù KL giảm rất mạnh, vì giáo viên FISA có biên quyết định nhỏ. Không có cấu hình nào đạt trung thành cân bằng 0,9. Đây là bằng chứng đo trên validation của root train, chưa phải outer test.

Nguồn mã khi chạy: commit `7a57fb2` (`source_dirty=false`). Mô hình vẫn thiếu `L_edge`, `L_A`, `L_B`, `L_rule`; chưa hiệu chỉnh FISA β/T hay chọn ngưỡng bằng inner validation theo bệnh nhân. Xem `run_manifest.json`, `tradeoff_results.json`, `table_tradeoff.csv`, `tradeoff_auc_fidelity.png` để có số chính xác và quan sát riêng từng fold/seed.
