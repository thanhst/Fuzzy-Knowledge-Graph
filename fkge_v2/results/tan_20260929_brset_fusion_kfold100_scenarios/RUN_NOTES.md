# FKG-E BRSET fusion 100 epoch: KB3 và KB5

Hai kịch bản hoàn tất trên 5 fold tách theo patient ID × 5 seed (42–46), 100 epoch, `quick=false`. Mỗi cấu hình có 25 quan sát fold × seed. SD là độ lệch chuẩn mẫu trên 25 quan sát; CI 95% là bootstrap 10.000 lần theo fold sau khi lấy trung bình seed trong fold.

KB3 quét đủ `d={8,16,32,64,128}`. AUC TB lần lượt là 0,9306; 0,9306; 0,9305; 0,9308; 0,9313, với SD khoảng 0,017. `d=8` được chọn theo quy tắc kích thước nhỏ khi AUC gần tương đương; số tham số TB khoảng 1.254 so với 20.070 ở `d=128`.

KB5 so sánh đồng xuất hiện toàn luật với cửa sổ `w=2`, cùng `K={2,5,10}`. AUC TB toàn luật là 0,9305–0,9306; cửa sổ là 0,9327–0,9328. Biên độ 0,0023 nhỏ so với SD 0,017–0,019. Sự gần như bất biến theo K chưa xác nhận H-E5 vì ablation vẫn đang kiểm tra đóng góp SGNS. Cấu hình chính theo định nghĩa mới là toàn luật; cửa sổ chỉ là đối chứng.

Nguồn mã khi chạy: commit `0c06757` (`source_dirty=false`). Tiến trình tổng được dừng sau khi KB5 ghi đủ kết quả để tránh chạy lặp KB4; các kịch bản còn lại có run ID riêng. Các thời gian trong bảng KB3 được đo lúc máy có tiến trình thực nghiệm khác, nên không dùng để kết luận chi phí; KB6 được chạy riêng.

Mô hình vẫn thiếu `L_edge`, `L_A`, `L_B`, `L_rule`; chưa hiệu chỉnh FISA β/T hay ngưỡng bằng inner validation theo bệnh nhân. Đây là validation của root train, chưa phải outer test. Chi tiết: `run_manifest.json`, `kb3_results.json`, `kb5_results.json`, `report_tong_hop.md`, `table_KB3.csv`, `table_KB5.csv`, `figures/`.
