# Baseline FKG-MM độc lập

Nhánh này là gói tối giản để phản biện chạy lại tầng FKG-MM trên đúng dữ liệu
FRB của thực nghiệm 5-fold theo bệnh nhân. Chạy:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
run.bat
```

Kết quả tham chiếu mới nhất: Accuracy `91.7 +/- 0.3`, F1 lớp bệnh
`16.6 +/- 12.9`, AUC-ROC `80.7 +/- 6.6`, Specificity `98.7 +/- 0.9`,
Sensitivity `11.4 +/- 10.2` (đơn vị phần trăm, mean +/- sample std của 5 fold).

Gói có sẵn rule data, ID bệnh nhân, feature map và dự đoán native để đối chiếu.
Nó kiểm chứng độc lập tầng FKG-MM từ FRB; không chứa ảnh fundus gốc và không
chạy lại bước trích xuất ảnh/FIS hay huấn luyện các baseline deep learning.
