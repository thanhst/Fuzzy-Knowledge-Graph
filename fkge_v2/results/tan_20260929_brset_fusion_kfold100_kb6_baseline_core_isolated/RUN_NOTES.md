# FKG-E BRSET fusion 100 epoch: KB6 và baseline

Chạy `quick=false`, 5 fold tách theo patient ID × 5 seed (42–46), 100 epoch cho FKG-E. SD là độ lệch chuẩn mẫu trên 25 quan sát fold × seed (FISA: 5 fold); CI 95% là bootstrap 10.000 lần theo fold sau khi trung bình seed.

## KB6: năm mức quy mô

| Số luật TB | FISA tuần tự ms/truy vấn TB ± SD | FISA bảng tra | FKG-E |
|---:|---:|---:|---:|
| 211 | 1,0909 ± 0,0866 | 0,0455 ± 0,0005 | 0,1456 ± 0,0283 |
| 422 | 2,0976 ± 0,1274 | 0,0458 ± 0,0006 | 0,1464 ± 0,0196 |
| 633 | 3,1088 ± 0,2043 | 0,0462 ± 0,0005 | 0,1507 ± 0,0016 |
| 844 | 4,1431 ± 0,2635 | 0,0463 ± 0,0003 | 0,1564 ± 0,0025 |
| 1.054,6 | 4,9030 ± 0,2480 | 0,0452 ± 0,0005 | 0,1628 ± 0,0068 |

Hệ số góc log–log trên năm điểm: FISA tuần tự 0,9453; bảng tra 0,0017; FKG-E 0,0662. Các kết quả phù hợp về xu hướng với chi phí quét luật của FISA tuần tự và chi phí gần hằng số ở quy mô này của bảng tra/FKG-E. Tại mức 100%, FKG-E nhanh hơn FISA tuần tự khoảng 30,1 lần, nhưng chậm hơn bảng tra khoảng 3,6 lần.

## Baseline trên cùng năm fold

| Phương pháp | AUC-ROC TB ± SD (CI 95%) | F1 TB ± SD | Thời gian huấn luyện TB |
|---|---:|---:|---:|
| FKG-E đầy đủ | 0,9305 ± 0,0170 (0,9149–0,9437) | 0,5142 ± 0,0560 | 9,975 s |
| MLP trên đặc trưng mờ | 0,9291 ± 0,0246 (0,9070–0,9488) | 0,5761 ± 0,0459 | 0,393 s |
| FISA bảng tra | 0,8231 ± 0,0791 (0,7568–0,8805) | 0,2845 ± 0,0875 | 0,006 s |

Chênh lệch AUC FKG-E–MLP chỉ 0,0014 và hai CI chồng lấp; không kết luận FKG-E tốt hơn MLP. MLP có F1 cao hơn và huấn luyện nhanh hơn khoảng 25 lần trong cấu hình này. Các baseline DeepWalk/TransE/DistMult hậu tố `-lite` chỉ kiểm tra luồng; Node2Vec chính thức chưa cài đặt, XGBoost và PyKEEN chưa có thư viện/kết quả trong môi trường. Xem JSON/CSV để có các hàng và chỉ số đầy đủ.

Nguồn mã khi chạy: commit `f830ea7` (`source_dirty=false`). Hai tiến trình dùng một luồng BLAS; KB4 được ghim vào logical CPU mask `1`, KB6/baseline vào mask `4`, cùng máy. Số đo thời gian được thu trong điều kiện này; chưa phải benchmark trên máy hoàn toàn rảnh. Mô hình vẫn thiếu `L_edge`, `L_A`, `L_B`, `L_rule`; chưa hiệu chỉnh FISA β/T hoặc ngưỡng theo inner validation. Đánh giá là validation của root train, chưa phải outer test.
