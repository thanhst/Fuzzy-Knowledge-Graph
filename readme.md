# Chẩn Đoán Bệnh Lý tiểu đường võng mạc bằng phương pháp FKGS

Hệ thống này sử dụng các phương pháp xử lý ảnh và học máy để phân tích ảnh fundus và dữ liệu liên quan nhằm hỗ trợ chẩn đoán các bệnh lý mắt, đặc biệt là các bệnh liên quan đến tiểu đường. Dự án kết hợp việc trích xuất đặc trưng từ ảnh fundus với dữ liệu bệnh nhân để đưa ra các phân tích và dự đoán chính xác.

## Mô Tả

Dự án này bao gồm các bước tiền xử lý ảnh, trích xuất đặc trưng bằng phương pháp GLCM (Gray Level Co-occurrence Matrix), và kết hợp dữ liệu metadata của bệnh nhân để huấn luyện mô hình học máy giúp chẩn đoán bệnh lý mắt - võng mạc tiểu đường.

Chúng tôi sử dụng thư viện OpenCV và scikit-image cho việc xử lý ảnh, kết hợp với các phương pháp như Fuzzy interfences system và Fuzzy knowledge graph, đặc biệt là Fuzzy knowledge graph sampling để đưa ra các dự đoán chính xác hơn, nhanh chóng hơn nhờ các mối quan hệ của các thuộc tính.

## Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.x
- Các thư viện: OpenCV, scikit-image, NumPy, Pandas
- Hệ điều hành: Windows

### Cài Đặt Các Phụ Thuộc

1. Clone repository này về máy:
   ```bash
   git clone https://github.com/thanhst/Fuzzy-Knowledge-Graph.git

Cài đặt các phụ thuộc:
    ```bash
    pip install -r requirements.txt

Cấu hình môi trường:

- Cài đặt Python 3.x

- Cài đặt các thư viện yêu cầu qua requirements.txt.

Cách Sử Dụng
- Tiền Xử Lý Dữ Liệu:
    - Để tiền xử lý ảnh fundus, chạy lệnh sau:

Cấu Trúc Thư Mục
Cấu trúc thư mục của dự án như sau:
```text
📦 Project
├── 📁 Source_code
│   ├── 📁 base # Đây là thư mục chứa các folder là cơ sở lý thuyết cho các phương pháp phát triển sau này.
│   ├── 📁 data # Đây là nơi chứa các dữ liệu như file tiền xử lý, file luật FRB và một số file mô hình.
        ├── 📁 BaseData # Đây là thư mục chứa các file thử nghiệm ban sơ.
        ├── 📁 Dataset # Đây là thư mục chứa các file dataset thử nghiệm
        ├── 📁 Dataset_diabetic # Đây là thư mục chứa các trường hợp chạy thử nghiệm tiền xử lý của bệnh võng mạc tiểu đường.
        ├── 📁 FIS # Đây là thư mục chứa input và output của thuật toán FIS.
        ├── 📁 FKG # Đây là thư mục chứa kết quả output của thuật toán FKG.
        ├── 📁 Metadata # Đây là thư mục chứa file metadata về y tế ban đầu chưa được xử lý và lựa chọn bệnh lý để chẩn đoán.
            └── 📄 Metadata.csv
    ├── 📁 main # Đây là thư mục chứa các kịch bản chạy chính của chương trình.
│   ├── 📁 models # Đây là thư mục chứa các kết quả model của từng kịch bản.
│   └── 📁 module # Đây là thư mục chứa các module được lập trình để phục vụ chương trình.
├── 📄 Scenario_diabetic_retinopathy_fusion_feature_with_glcm.bat # Đây là file bat chạy kịch bản kết hợp thuộc tính GLCM của ảnh với metadata dạng table.
├── 📄 Scenario_diabetic_retinopathy_fusion_feature_with_statistical.bat # Đây là file bat chạy kịch bản kết hợp thuộc tính statistical của ảnh với metadata dạng table.
├── 📄 Scenario_diabetic_retinopathy_fusion_feature.bat # Đây là file bat chạy kịch bản kết hợp thuộc tính GLCM, statistical của ảnh với metadata dạng table.
├── 📄 Scenario_diabetic_retinopathy_GLCM_feature.bat # Đây là file bat chạy kịch bản thuộc tính GLCM của ảnh.
├── 📄 Scenario_diabetic_retinopathy_image_feature.bat # Đây là file bat chạy kịch bản thuộc tính ảnh.
├── 📄 Scenario_diabetic_retinopathy_statistical_feature.bat # Đây là file bat chạy kịch bản statistical của ảnh.
├── 📄 Scenario_diabetic_retinopathy_table_feature.bat # Đây là file bat chạy kịch barn metadata dạng table.
└── 📄 README.md

