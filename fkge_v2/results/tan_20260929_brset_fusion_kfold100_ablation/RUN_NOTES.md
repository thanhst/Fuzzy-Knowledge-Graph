# FKG-E BRSET fusion 100 epoch: ablation

Chín biến thể, mỗi biến thể 5 fold tách theo patient ID × 5 seed (42–46), 100 epoch, `quick=false`. SD là độ lệch chuẩn mẫu của 25 quan sát; CI 95% là bootstrap 10.000 lần theo fold sau khi lấy trung bình seed trong fold.

| Biến thể | AUC-ROC TB ± SD (CI 95%) | BalAcc TB ± SD |
|---|---:|---:|
| FKG-E đầy đủ | 0,9305 ± 0,0170 (0,9149–0,9437) | 0,8411 ± 0,0408 |
| Chỉ `L_pred` | 0,9303 ± 0,0176 (0,9139–0,9433) | 0,8366 ± 0,0412 |
| Bỏ `L_pred` | 0,6206 ± 0,2096 (0,5889–0,6744) | 0,5621 ± 0,0958 |
| Chỉ SGNS | 0,5980 ± 0,1932 (0,5585–0,6541) | 0,5302 ± 0,0785 |
| Bỏ SGNS | 0,9306 ± 0,0170 (0,9150–0,9437) | 0,8407 ± 0,0407 |
| Bỏ loss nút | 0,9305 ± 0,0184 (0,9137–0,9451) | 0,8369 ± 0,0382 |
| Bỏ chưng cất FISA | 0,9300 ± 0,0155 (0,9158–0,9416) | 0,8354 ± 0,0437 |
| Bỏ L2 | 0,9305 ± 0,0170 (0,9149–0,9437) | 0,8411 ± 0,0408 |
| Gộp trung bình đều | 0,9231 ± 0,0205 (0,9081–0,9381) | 0,8445 ± 0,0219 |

Với mã hiện tại, chất lượng chủ yếu đến từ `L_pred`. Các thành phần cấu trúc đã cài đặt và chưng cất chưa cho mức tăng AUC rõ ràng. Đây là kết quả mô tả trên validation của root train; chưa có kiểm định trên outer test.

Nguồn mã khi chạy: commit `7a57fb2` (`source_dirty=false`). Mô hình vẫn thiếu `L_edge`, `L_A`, `L_B`, `L_rule`; chưa hiệu chỉnh giáo viên FISA β/T hoặc chọn ngưỡng trên inner validation theo bệnh nhân. Xem `ablation_results.json` để có toàn bộ chỉ số, CI và 225 quan sát riêng theo fold/seed.
