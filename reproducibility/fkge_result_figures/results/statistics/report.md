# Kết quả FKG-E trên BRSET

Báo cáo sử dụng bản FKG-E hiện tại. Kết quả được tái lập từ 5 cấu hình đã chọn; không chạy lại quá trình chọn 108 cấu hình/fold. Seed 42, tối đa 200 vòng, early stopping theo nguồn.

## 1. Vì sao mỗi fold chỉ có 583–584 bệnh nhân train?

**Cột bệnh nhân là số bệnh nhân trong tập train của từng fold. Toàn bộ phần dữ liệu được chia fold ở thực nghiệm này gồm 1.208 ảnh của 729 bệnh nhân.**

| Phạm vi | Số ảnh | Số bệnh nhân |
| --- | --- | --- |
| Bảng nhãn BRSET đầy đủ | 16266 | 8524 |
| Ảnh khớp nhãn trong lần xuất dữ liệu | 1529 | 897 |
| Tập test giữ riêng từ đầu | 321 | 168 |
| Phần dữ liệu được chia 5 fold | 1208 | 729 |

Code xuất dữ liệu quét các file ảnh trong thư mục đầu vào rồi ghép với bảng nhãn. Ở lần xuất này, thư mục có 1.529 ảnh khớp nhãn của 897 bệnh nhân. Mặc dù bảng nhãn chứa 16.266 ảnh của 8.524 bệnh nhân, số ảnh đầu vào thực sự chỉ là một subset. Sự thu hẹp phạm vi đã xảy ra ở bước lấy ảnh đầu vào, trước khi chia 5 fold.

Trong 897 bệnh nhân đó, 168 bệnh nhân với 321 ảnh được giữ riêng làm root test. Phần còn lại có **729 bệnh nhân với 1.208 ảnh** và mới được chia thành 5 fold. Tập root test giữ riêng không được đánh giá trong bảng kết quả 5 fold này.

| Fold | Ảnh train trước SMOTE | Bệnh nhân train | Ảnh giữ lại | Bệnh nhân giữ lại |
| --- | --- | --- | --- | --- |
| fold_01 | 966 | 583 | 242 | 146 |
| fold_02 | 966 | 583 | 242 | 146 |
| fold_03 | 966 | 583 | 242 | 146 |
| fold_04 | 967 | 583 | 241 | 146 |
| fold_05 | 967 | 584 | 241 | 145 |

Ví dụ fold 1: **583 + 146 = 729 bệnh nhân**, tương ứng **966 + 242 = 1.208 ảnh**. Mỗi lượt dùng bốn phần để train, nên số bệnh nhân train xấp xỉ `729 × 4/5 = 583,2`; bốn fold có 583 bệnh nhân train và một fold có 584. Một bệnh nhân có thể có nhiều ảnh, nên số ảnh không bằng số bệnh nhân. Không cộng số bệnh nhân train của năm lượt: mỗi bệnh nhân được dùng để train ở bốn lượt.

Sau SMOTE, fold 1 có **966 + 812 = 1.778 dòng huấn luyện/luật**. 812 dòng bổ sung là mẫu tổng hợp; số bệnh nhân train thực vẫn là **583**.

Nếu chia 5 fold trực tiếp trên toàn bộ 8.524 bệnh nhân thì mỗi lượt train sẽ khoảng 6.800 bệnh nhân. Kết quả hiện tại cần được ghi là thực nghiệm trên subset BRSET. Để báo cáo trên toàn bộ BRSET, cần đưa đủ ảnh vào pipeline, tạo lại manifest/fold, đặc trưng, mờ hóa, cân bằng, luật và chạy lại mô hình. Nạp lại các file fold/đặc trưng đã xuất sẵn trên Colab vẫn giữ nguyên quy mô subset.

![Số bệnh nhân theo fold](new_zip/patient_counts.png)

[CSV phạm vi dữ liệu](new_zip/table_population.csv) · [CSV chi tiết từng fold](new_zip/table_split.csv) · [Bằng chứng số lượng](../../population.json)

## 2. Kết quả chính FISA và FKG-E tuned

Mỗi ô là trung bình ± SD mẫu qua **5 fold** (`ddof=1`), với một seed. Không coi 50 lần đo thời gian là 50 mẫu độc lập. Các fold dùng chung một phần dữ liệu huấn luyện, nên SD này chỉ mô tả độ biến thiên, không khẳng định ý nghĩa thống kê.

| Độ đo | FISA | FKG-E | Δ (FKG-E − FISA) | Fold FKG-E hơn |
| --- | --- | --- | --- | --- |
| ROC-AUC | 0.8697 ± 0.0576 | 0.9215 ± 0.0182 | +0.0518 | 3/5 |
| PR-AUC (AP) | 0.5231 ± 0.0691 | 0.5736 ± 0.0587 | +0.0505 | 3/5 |
| Balanced Accuracy | 0.7743 ± 0.0554 | 0.7554 ± 0.0529 | -0.0189 | 2/5 |
| Độ nhạy | 0.8121 ± 0.1773 | 0.7211 ± 0.2391 | -0.0911 | 1/5 |
| Độ đặc hiệu | 0.7364 ± 0.1298 | 0.7897 ± 0.2434 | +0.0533 | 4/5 |
| F1 lớp dương | 0.3537 ± 0.0716 | 0.4273 ± 0.1404 | +0.0735 | 4/5 |
| Accuracy | 0.7427 ± 0.1086 | 0.7842 ± 0.2071 | +0.0414 | 4/5 |

PR-AUC là Average Precision theo code nguồn. F1 ở bảng này là **F1 lớp dương**. Các chỉ số dùng ngưỡng của mô hình nguồn; AUC/AP dùng điểm liên tục. AUC/AP tốt hơn không đồng nghĩa mọi chỉ số phân loại tốt hơn.

![Bảng kết quả FKG-E](new_zip/result_summary.png)

[CSV kết quả chính](new_zip/table_main.csv) · [Kiểm tra tái lập 70 chỉ số](new_zip/table_replay_checks.csv)

Chênh lệch lớn nhất khi tái lập 7 chỉ số × 2 mô hình × 5 fold so với kết quả nguồn: `0`. Các con số phản ánh subset **1.208 ảnh / 729 bệnh nhân**. Inner validation vẫn dùng dữ liệu đã tiền xử lý/SMOTE và đồ thị nguồn; bảng này không chứng nhận đã khắc phục yêu cầu validation của phản biện.

## 3. Cấu hình và từng fold

| Fold | α | d | λ_I | λ_P | Loss val | AUC val | AP val | AUC test | AP test | Vòng |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_01 | 0.3000 | 32 | 0.1000 | 0.5000 | 0.4983 | 0.8504 | 0.7584 | 0.9375 | 0.6092 | 41 |
| fold_02 | 0.5000 | 16 | 0.1000 | 0.3000 | 0.2625 | 0.9621 | 0.9385 | 0.9363 | 0.6226 | 23 |
| fold_03 | 0.3000 | 64 | 0.1000 | 0.3000 | 0.6084 | 0.7618 | 0.7516 | 0.8967 | 0.4808 | 31 |
| fold_04 | 0.3000 | 32 | 0.3000 | 0.5000 | 0.3108 | 0.9473 | 0.9192 | 0.9291 | 0.5507 | 33 |
| fold_05 | 0.3000 | 16 | 0.5000 | 0.7000 | 0.1960 | 0.9977 | 0.9973 | 0.9080 | 0.6047 | 46 |

[CSV cấu hình](new_zip/table_selected_folds.csv) · [Chỉ số từng fold và mô hình](new_zip/table_fold_metrics.csv) · [Toàn bộ lưới đã lưu](new_zip/table_archived_grid.csv)

## 4. Chia dữ liệu và thông số chạy

| Fold | Train trước SMOTE | Train sau SMOTE / luật | Bệnh nhân train | SMOTE | Học / val | Test (dương) |
| --- | --- | --- | --- | --- | --- | --- |
| fold_01 | 966 | 1778 | 583 | 812 | 1421 / 357 | 242 (20) |
| fold_02 | 966 | 1776 | 583 | 810 | 1421 / 355 | 242 (19) |
| fold_03 | 966 | 1776 | 583 | 810 | 1420 / 356 | 242 (19) |
| fold_04 | 967 | 1780 | 583 | 813 | 1424 / 356 | 241 (20) |
| fold_05 | 967 | 1778 | 584 | 811 | 1422 / 356 | 241 (19) |

| Chỉ số phụ của 5 cấu hình đã chọn | Trung bình ± SD |
| --- | --- |
| Đồng thuận nhãn FISA | 0.7800 ± 0.1201 |
| Dev căn cứ luật | 0.5097 ± 0.0785 |
| FISA bảng tra (ms/truy vấn, replay) | 1.2003 ± 0.0764 |
| FISA tuần tự (ms/truy vấn, replay) | 35.6372 ± 0.6150 |
| FKG-E (ms/truy vấn, replay) | 0.2667 ± 0.0378 |
| Huấn luyện FKG-E (s, replay) | 82.4647 ± 18.9715 |
| Huấn luyện FKG-E (s, tác giả lưu) | 33.1691 ± 14.8484 |

Thời gian phụ thuộc máy và tải khi chạy hai worker; không gộp thời gian tác giả với replay để tính speedup. Số luật là số dòng sau SMOTE, không phải số bệnh nhân. [CSV chia dữ liệu](new_zip/table_split.csv) · [Ngưỡng, T, gradient, dừng và thời gian](new_zip/table_auxiliary.csv)

## 5. CI/p-value theo công thức trong code nguồn (tham khảo)

Mục này chỉ tái hiện công thức trong `fkge_branch/report/report_nested.py`: CI t có 4 bậc tự do; p NB dùng tỉ lệ test/train **sau SMOTE**. Tỉ lệ đó khác số ảnh train gốc (test/train trước SMOTE = 0,25). CI có thể vượt [0;1]. Không dùng các cột này để khẳng định ý nghĩa thống kê hay xem 5 fold là các thí nghiệm độc lập.

| Độ đo | CI95 FKG-E theo code gốc | p NB theo code gốc |
| --- | --- | --- |
| auc | [0.8989; 0.9441] | 0.202 |
| pr_auc | [0.5008; 0.6464] | 0.462 |
| balanced_accuracy | [0.6897; 0.8211] | 0.714 |
| sensitivity | [0.4242; 1.0179] | 0.354 |
| specificity | [0.4875; 1.0919] | 0.727 |
| f1_pos | [0.2530; 0.6015] | 0.493 |
| accuracy | [0.5270; 1.0413] | 0.759 |

[CSV công thức nguồn](new_zip/table_reference_inference.csv)

## 6. Phạm vi kết quả

Báo cáo này trình bày kết quả tuned của năm cấu hình đã chọn trên validation. Các thí nghiệm KB2–KB6, ablation và baseline khác cần được chạy với pipeline hiện tại trước khi có thể báo cáo như kết quả của phiên bản này. Không chọn cấu hình theo chỉ số test cao nhất. Xem `statistics_manifest.json` để đối chiếu SHA256 nguồn, script và các file được xuất.
