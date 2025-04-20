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
├── 📁 Source_code                       # Mã nguồn chính
│   ├── 📁 base                          # Thư mục chứa các lý thuyết cơ sở cho phương pháp phát triển sau này.
│   ├── 📁 data                          # Dữ liệu như file tiền xử lý, luật FRB, và các file mô hình
│   │   ├── 📁 BaseData                  # File thử nghiệm ban sơ
│   │   ├── 📁 Dataset                   # File dataset thử nghiệm ban sơ
│   │   ├── 📁 Dataset_diabetic          # Thử nghiệm tiền xử lý bệnh võng mạc tiểu đường của từng kịch bản chạy
│   │   ├── 📁 FIS
│   │   │   ├── 📁 input                 # input là thư mục chứa các input đầu vào train, test của mô hình FIS với từng kịch bản
|   |   |   ├── 📁 output                # ouput là thư mục chứa các output đầu ra của FIS là FRB cơ bản dùng cho FKG, rules list, ...
│   │   ├── 📁 FKG                       # Output của thuật toán FKG
│   │   └── 📁 Metadata                  # Metadata về y tế ban đầu
│   │       └── 📄 Metadata.csv
│   ├── 📁 main                          # Các kịch bản chạy chính của chương trình
│   ├── 📁 models                        # Kết quả mô hình của từng kịch bản
│   └── 📁 module                        # Các module phục vụ chương trình
│       ├── 📁 Convert                   # Bộ chuyển đổi luật dạng số học thành dạng ngôn ngữ
│       ├── 📁 FCM
│       │   ├── 📄 FCM_Function.py       # Thuật toán phân cụm mờ
│       │   └── 📄 standardize_data.py   # Code chuẩn hóa dữ liệu
│       ├── 📁 FIS
│       │   └── 📄 FIS.py                # Thuật toán FIS
│       ├── 📁 FKG
│       │   ├── 📄 FKG_general.py        # Thuật toán FKG dùng cho bài toán nhiều nhãn
│       │   ├── 📄 FKG_model.py          # Thuật toán FKG dùng cho bài toán 6 nhãn
│       │   ├── 📄 FKG_S.py              # Thuật toán FKG sampling
│       │   └── 📄 FKG.py                # Thuật toán FKG cơ bản cho hai nhãn
│       ├── 📁 Helper                    # Một số hàm hỗ trợ
│       ├── 📁 Membership_Function
│       │   ├── 📄 ExpMF.py              # Mức độ hàm thành viên tính bằng hàm mũ
│       │   ├── 📄 GaussMF.py            # Mức độ hàm thành viên tính bằng hàm Gaussian
│       │   ├── 📄 SigmoidMF.py          # Mức độ hàm thành viên tính bằng hàm Sigmoid
│       │   ├── 📄 TrapezoidalMF.py      # Mức độ hàm thành viên tính bằng hàm hình thang
│       │   └── 📄 TriangleMF.py         # Mức độ hàm thành viên tính bằng hàm tam giác
│       ├── 📁 Module_CPP                # Các module c++ phục vụ tính toán
│       ├── 📁 Processing_Data           # Code tiền xử lý cho các kịch bản
│       ├── 📁 Rules_Function
│       │   ├── 📄 Rules_gen.py          # Code sinh luật mờ
│       │   ├── 📄 Rules_reduce.py       # Code giảm các luật conflict và các luật trùng nhau
│       │   └── 📄 RuleWeight.py         # Code tính trọng số luật mờ
│       └── 📁 Setup_module              # Chứa thư mục CMAKE và các file c++, pyd phục vụ cho việc xây dựng module python bằng c++ hỗ trợ hiệu suất tính toán.
├── 📄 Scenario_diabetic_retinopathy_fusion_feature_with_glcm.bat                   # Kịch bản kết hợp GLCM với metadata
├── 📄 Scenario_diabetic_retinopathy_fusion_feature_with_statistical.bat            # Kịch bản kết hợp statistical với metadata
├── 📄 Scenario_diabetic_retinopathy_fusion_feature.bat                             # Kết hợp GLCM, statistical với metadata
├── 📄 Scenario_diabetic_retinopathy_GLCM_feature.bat                               # Kịch bản thuộc tính GLCM của ảnh
├── 📄 Scenario_diabetic_retinopathy_image_feature.bat                              # Kịch bản thuộc tính ảnh
├── 📄 Scenario_diabetic_retinopathy_statistical_feature.bat                        # Kịch bản statistical của ảnh
├── 📄 Scenario_diabetic_retinopathy_table_feature.bat                              # Kịch bản metadata dạng table
└── 📄 README.md                                                                    # Tệp README của dự án


