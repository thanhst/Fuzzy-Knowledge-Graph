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

2. Cài đặt các phụ thuộc:
   ```bash
   pip install -r requirements.txt

3. Cấu hình môi trường:

- Cài đặt Python 3.x

- Cài đặt các thư viện yêu cầu qua requirements.txt.

4. Cách Sử Dụng:

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
```
### Hướng dẫn chạy chương trình.
- Trong folder chính của dự án, có các file batch khi thực hiện trích các đặc trưng từ ảnh, bảng và kết hợp. Bấm chạy các file batch để chương trình hoạt động.
- Trong folder fusion-case là các trường hợp fusion của hai mô thức ảnh và bảng, có các trường hợp như kết hợp fusion theo wrapper, feature selection, filter multimodal , hadamard và tensor selection. Bấm chạy các file batch để chương trình hoạt động.
- Trong folder Multimodality là các trường hợp fusion của 3 mô thức, gồm ảnh võng mạc đáy mắt ( fundus ) và ảnh chụp cắt lớp của mắt ( OCT B-Scan), file dữ liệu bổ trợ dạng bảng. Có 4 trường hợp kết hợp như sau: Fundus + Table, Fundus + OCT, OCT+ Table, Fundus + OCT + Table. Trong dữ liệu dạng bảng có các thuộc tính như sau : race,male,hispanic,maritalstatus,language,dr_class. Dữ liệu OCT images được trích ra các thuộc tính : Contrast Feature,Dissimilarity Feature,Homogeneity Feature,Energy Feature,Correlation Feature,ASM Feature,Mean Feature,Variance Feature,Standard Deviation Feature,RMS Feature,dr_class. Dữ liệu fundus images cũng chứa các thuộc tính tương tự OCT images. Từ đó, sử dụng các phương pháp kết hợp như Feature Selection, Filter Multimodal Selection, Hadamard Selection, Tensor Selection và Wrapper Selection để kết hợp dữ liệu với việc trích ra 6 đặc trưng tốt nhất trong 10 đặc trưng của fundus images, 6 đặc trưng tốt nhất trong 10 đặc trưng của OCT images và 4 đặc trưng của dữ liệu bổ trợ dạng bảng. Sau các phương pháp thì nhận thấy phương pháp Feature Selection phù hợp với trích đặc trưng từ các thuộc tính của bộ dữ liệu này, mang lại độ chính xác cao nhất trong 5 phương pháp kết hợp kể trên.
<br>
*** Lưu ý: Chương trình chạy code với tính toán FISA bằng module C++ được build trên window, máy linux sẽ chưa thể hoạt động được. ***


