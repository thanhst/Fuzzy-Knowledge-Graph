import os
import cv2
import numpy as np
import pandas as pd
from skimage import color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops

# Tạo DataFrame rỗng với các cột đặc trưng
df = pd.DataFrame(
    columns=[
        "Contrast Feature",
        "Dissimilarity Feature",
        "Homogeneity Feature",
        "Energy Feature",
        "Correlation Feature",
        "ASM Feature",
    ]
)

# Khởi tạo các list để lưu trữ đường dẫn ảnh và tên ảnh
matrix1 = []
list_of_images = []
name_of_images = []

# Lấy đường dẫn của các ảnh từ 3 thư mục
for i in range(1, 4):
    path_of_images = f"/kaggle/input/skin-cancer/imgs_part_{i}/imgs_part_{i}"
    images = os.listdir(path_of_images)
    name_of_images.extend(images)
    list_of_images.extend([os.path.join(path_of_images, img) for img in images])
    
# Tính ma trận GLCM cho từng ảnh
for image_path in list_of_images:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: could not read {image_path}")
        continue
    gray = color.rgb2gray(img)
    image_ubyte = img_as_ubyte(gray)  # Chuyển sang ảnh 8-bit

    # Giảm số mức xám từ 256 xuống 16 mức (giống như nén dữ liệu)
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
    inds = np.digitize(image_ubyte, bins)
    
    max_value = inds.max() + 1
    matrix_coocurrence = graycomatrix(
        inds,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=max_value,
        normed=False,
        symmetric=False,
    )
    matrix1.append(matrix_coocurrence)

# Định nghĩa các hàm tính đặc trưng (đưa ra trung bình qua các hướng)
def contrast_feature(matrix):
    return np.mean(graycoprops(matrix, 'contrast'))

def dissimilarity_feature(matrix):
    return np.mean(graycoprops(matrix, 'dissimilarity'))

def homogeneity_feature(matrix):
    return np.mean(graycoprops(matrix, 'homogeneity'))

def energy_feature(matrix):
    return np.mean(graycoprops(matrix, 'energy'))

def correlation_feature(matrix):
    return np.mean(graycoprops(matrix, 'correlation'))

def asm_feature(matrix):
    return np.mean(graycoprops(matrix, 'ASM'))

# Tính các đặc trưng cho mỗi ma trận và lưu vào các list
CF = []
DF = []
HF = []
EF = []
COR = []
ASM_values = []

for matrix in matrix1:
    CF.append(contrast_feature(matrix))
    DF.append(dissimilarity_feature(matrix))
    HF.append(homogeneity_feature(matrix))
    EF.append(energy_feature(matrix))
    COR.append(correlation_feature(matrix))
    ASM_values.append(asm_feature(matrix))

# Gán các giá trị đặc trưng vào DataFrame
df["Contrast Feature"] = CF
df["Dissimilarity Feature"] = DF
df["Homogeneity Feature"] = HF
df["Energy Feature"] = EF
df["Correlation Feature"] = COR
df["ASM Feature"] = ASM_values

# Đảm bảo số lượng index phù hợp với số ảnh đã xử lý
df.index = name_of_images[:len(df)]
df.index.name = "img_id"

df
