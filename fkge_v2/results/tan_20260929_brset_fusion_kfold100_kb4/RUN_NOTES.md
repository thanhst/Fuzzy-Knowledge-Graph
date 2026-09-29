# FKG-E BRSET fusion 100 epoch: KB4 độ nhạy trọng số

Chạy đủ 30 cấu hình: năm trọng số đã cài đặt (`λ_S`, `λ_N`, `λ_I`, `λ_P`, `λ_C`) × sáu hệ số nhân `{0; 0,1; 0,3; 1; 3; 10}`. Mỗi cấu hình có 5 fold theo patient ID × 5 seed (42–46), 100 epoch, `quick=false`, tức 25 quan sát. SD là độ lệch chuẩn mẫu của 25 quan sát; CI 95% là bootstrap 10.000 lần theo fold sau khi trung bình seed trong fold.

| Cấu hình | AUC-ROC TB ± SD (CI 95%) | Đồng thuận TB | Trung thành cân bằng TB | κ TB |
|---|---:|---:|---:|---:|
| Mặc định: λ_I=2, λ_P=20, λ_C=1e-5 | 0,9305 ± 0,0170 (0,9149–0,9437) | 0,7033 | 0,6685 | 0,3498 |
| λ_P=0 | 0,6206 ± 0,2096 (0,5889–0,6744) | 0,5976 | 0,5480 | 0,1021 |
| λ_P=6 (×0,3) | 0,9226 ± 0,0258 (0,9024–0,9446) | 0,7149 | 0,6901 | 0,3798 |
| λ_I=0 | 0,9300 ± 0,0155 (0,9158–0,9416) | 0,6999 | 0,6633 | 0,3413 |
| λ_I=20 (×10) | 0,9311 ± 0,0203 (0,9132–0,9482) | 0,7098 | 0,6791 | 0,3696 |
| λ_C=0 | 0,930517 ± 0,017024 (0,914863–0,943675) | 0,7033 | 0,6685 | 0,3498 |
| λ_C=1e-4 (×10) | 0,930544 ± 0,017078 (0,914833–0,943731) | 0,7035 | 0,6686 | 0,3500 |

Nhóm λ_P có ảnh hưởng lớn nhất: bỏ dự đoán có nhãn làm AUC giảm khoảng 0,31. λ_P=6 cho trung thành cân bằng cao hơn mặc định 0,0217 nhưng AUC thấp hơn 0,0079; đây chỉ là điểm đánh đổi mô tả trên validation. Tăng λ_I từ 0 lên 20 cải thiện trung thành cân bằng 0,6633→0,6791, vẫn xa mục tiêu 0,9. Trong dải 0–1e-4, λ_C làm AUC thay đổi dưới 0,00003; không có bằng chứng về cải thiện chất lượng từ L2 trong dải thử này.

Nguồn mã lúc chạy: commit `7a57fb2` (`source_dirty=false`). Bốn trọng số `λ_E`, `λ_A`, `λ_B`, `λ_R` tương ứng `L_edge`, `L_A`, `L_B`, `L_rule` chưa triển khai nên không có hàng thực nghiệm cho chúng. Giáo viên FISA β/T và ngưỡng quyết định chưa được hiệu chỉnh bằng inner validation theo bệnh nhân. Đây là validation của root train, chưa phải outer test. Xem `kb4_results.json` và `table_KB4.csv` để có cả 30 cấu hình và CI/quan sát từng fold/seed.
