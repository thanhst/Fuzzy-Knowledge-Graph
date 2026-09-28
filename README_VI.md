# FKG-MM chạy đầy đủ

Nhánh `MM` chạy lại toàn bộ chuỗi thực nghiệm FKG-MM từ bảng đặc trưng liên tục:
ghép ID bệnh nhân, kiểm tra chống rò rỉ, chuẩn hóa theo train fold, chọn 7 đặc
trưng ảnh + 9 đặc trưng bảng, BorderlineSMOTE, FCM/FIS sinh luật mờ, huấn luyện
FKG-MM và đánh giá 5 fold.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
run.bat
```

Lệnh chạy còn đối chiếu luật và dự đoán mới sinh với bản native đã lưu. Kết quả
đúng sẽ in `reference rules match: True` và
`reference predictions match: True`. Dữ liệu đặc trưng, ID, split 5 fold và dữ
liệu tham chiếu đều có sẵn trong nhánh; không cần build C++/CUDA.
