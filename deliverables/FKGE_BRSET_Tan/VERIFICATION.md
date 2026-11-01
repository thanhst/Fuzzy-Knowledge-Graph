# Kiểm tra gói trước khi bàn giao

- 6 bài kiểm tra đã đạt: giá trị/attribution FISA; loss/gradient FKG-E so với code
  zip nguồn trên luật trùng có trọng số; bất biến theo block; SMOTE giống thư viện
  và giữ đúng hai mẫu cha; teacher không tách bệnh nhân/nguồn SMOTE; validation
  không thay tham số tiền xử lý. Một bài có thể kiểm tra nhiều tính chất.
- Đã chạy chuẩn bị trên đủ 16.266 ảnh, cả 5 fold; bảng số lượng trong
  `CHECKED_FOLD_COUNTS.csv` và `.png`. Không có bệnh nhân giao nhau giữa
  inner train, validation và outer test.
- Đã chạy thử toàn bộ luồng CPU trên dữ liệu tổng hợp: 200 ảnh / 100 bệnh nhân,
  một outer fold, hai cấu hình, tối đa hai epoch; tạo được checkpoint, dự đoán,
  bảng chỉ số và ảnh. Đây là kiểm tra phần mềm, không phải chỉ số BRSET.
- Chưa huấn luyện đủ lưới trên bộ BRSET thật; chưa xác nhận lượt chạy GPU/Colab
  đầy đủ. Môi trường kiểm thử và lệnh chạy nằm trong README; lượt chạy mới tự ghi
  môi trường thực tế, phần cứng, câu lệnh, seed, commit và hash vào output.
