# Chẩn Đoán Bệnh Lý Võng Mạc Tiểu Đường bằng Phương Pháp FKGS

Hệ thống sử dụng các phương pháp xử lý ảnh và học máy để phân tích ảnh fundus cùng dữ liệu metadata hỗ trợ chẩn đoán các bệnh lý mắt, đặc biệt là bệnh võng mạc do tiểu đường. Phương pháp này kết hợp trích xuất đặc trưng GLCM từ ảnh với thông tin y tế nhằm nâng cao độ chính xác chẩn đoán.

## 📌 Mô Tả

Dự án gồm các bước:

* Tiền xử lý ảnh.
* Trích xuất đặc trưng bằng GLCM (Gray Level Co-occurrence Matrix).
* Kết hợp với dữ liệu bệnh nhân.
* Huấn luyện mô hình học máy với thuật toán Fuzzy Knowledge Graph Sampling (FKGS).

Sử dụng các thư viện như OpenCV, scikit-image, NumPy và Pandas. Các thuật toán fuzzy như FIS (Fuzzy Inference System), FKG và FRB (Fuzzy Rule Base) được tích hợp để xử lý các mối quan hệ phức tạp trong dữ liệu.

## ⚙️ Cài Đặt

### Yêu Cầu Hệ Thống

* Python 3.x
* Thư viện: OpenCV, scikit-image, NumPy, Pandas
* Hệ điều hành: **Windows** *(hiện tại chưa hỗ trợ Linux do có module C++ build sẵn)*

### Cài Đặt Các Phụ Thuộc

```bash
# Clone repo
git clone https://github.com/thanhst/Fuzzy-Knowledge-Graph.git
cd Fuzzy-Knowledge-Graph

# Cài đặt thư viện
pip install -r requirements.txt
```

## 📁 Cấu Trúc Thư Mục

```text
📦 Project
├── 📁 Source_code
│   ├── base                        # Lý thuyết nền
│   ├── data                        # Dữ liệu GLCM, metadata, luật FRB
│   │   ├── BaseData
│   │   ├── Dataset
│   │   ├── Dataset_diabetic
│   │   ├── FIS
│   │   │   ├── input              # Input train/test cho FIS
│   │   │   └── output             # Output là FRB, rules list
│   │   ├── FKG                    # Output của thuật toán FKG
│   │   └── Metadata
│   │       └── Metadata.csv
│   ├── main                        # Các kịch bản chạy chính
│   ├── models                      # Kết quả huấn luyện mô hình
│   └── module                      # Các module xử lý
│       ├── Convert
│       ├── FCM
│       ├── FIS
│       ├── FKG
│       ├── Helper
│       ├── Membership_Function
│       ├── Module_CPP             # Mã nguồn C++ build module
│       ├── Processing_Data
│       ├── Rules_Function
│       └── Setup_module
├── 📄 Các file *.bat chạy các kịch bản tiền xử lý và huấn luyện
└── 📄 README.md
```

## ▶️ Hướng Dẫn Chạy Chương Trình

### 1. Chạy Các Kịch Bản

Trong thư mục chính có các file `.bat` như:

* `Scenario_diabetic_retinopathy_image_feature.bat`
* `Scenario_diabetic_retinopathy_GLCM_feature.bat`
* `Scenario_diabetic_retinopathy_fusion_feature.bat`
* ...

Chạy file phù hợp với kịch bản bạn muốn thực hiện.
Nếu bạn muốn lập trình thêm (tạo các kịch bản code từ FKG) thì có thể thực hiện theo dạng file code như sau:
```
import sys
import os

# Lấy đường dẫn tuyệt đối tới thư mục gốc của project (ở đây là Source_code)
# Bạn nên đưa file code vào thư mục main và sau đó tạo một file bat tương tự như ví dụ bat ở trên để dễ dàng chạy nhé!
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # lên 2 cấp

if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from module.FIS.FIS import FIS
from module.FKG.FKG_general import FKG
from module.FKG.FKG_S import FKGS


print("Diabetic Retinopathy Fusion Feature")

print("__________Running Processing___________")
# Đây là file code bạn tiền xử lý cho dữ liệu của bạn
from module.Processing_Data import Diabetic_fusion_processing

print("__________Running FIS___________")
# Sử dụng FIS cho việc tiền đánh giá dữ liệu gốc bằng luật mờ, đồng thời cũng tạo ra một FRB dùng cho FKG
FIS(fileName="Diabetic Retinopathy Feature",
    filePath=os.path.join(project_root,"data\Dataset_diabetic\Fusion_feature\data_process.csv"),
    cluster=[3,2,4,2,2,2,2,2,2,2,2,2,2,5,5,5,5,5,5,5,5,5,5,2])
print("--------------------------------")

print("__________Running FKG___________")
# Ở đây chỉ cần gọi thư viện và truyền các tham số, dữ liệu là hoàn toàn FKG có thể hoạt động được.
traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Feature/FRB/TrainDataRule_All.csv'))
testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Feature/FRB/TestDataRule.csv'))
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Feature")
print("--------------------------------")

print("__________Running FKG-S___________")
traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Feature/FRB/TrainDataRule_All.csv'))
testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Feature/FRB/TestDataRule.csv'))
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKGS()
fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Feature",ran=20,e=0.2,folderPath=project_root)
print("-"*100)
```
### 2. Kết Hợp Dữ Liệu (Fusion)

* **Thư mục `fusion-case`**: chứa các kịch bản kết hợp 2 mô thức (ảnh + bảng).
* **Thư mục `Multimodality`**: chứa các kịch bản kết hợp 3 mô thức: Fundus, OCT, và metadata dạng bảng.

Các phương pháp kết hợp:

* Feature Selection
* Filter Multimodal
* Hadamard
* Tensor Selection
* Wrapper

> Qua thực nghiệm, phương pháp **Feature Selection** cho kết quả chính xác cao nhất.

### 3. Quy Trình Xử Lý

1. Tiền xử lý dữ liệu
2. Chọn phương pháp kết hợp
3. Trích xuất đặc trưng
4. Sinh luật FRB bằng thuật toán FCM
5. Huấn luyện mô hình FKGS với các tham số `ran`, `e`: (15, 0.2), (15, 0.3), (20, 0.2), (20, 0.3)
6. Kiểm thử mô hình với tập test

⚠️ **Lưu ý:** Mô đun C++ tính toán FIS chỉ hỗ trợ trên Windows!
