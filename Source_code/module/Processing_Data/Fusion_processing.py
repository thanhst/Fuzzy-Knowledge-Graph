import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import os
import cv2,csv
from pathlib import Path
base_path = Path(__file__).resolve().parents[2]
import time
start = time.time()
# Làm rõ vùng tối/sáng, giúp mạch máu và tổn thương dễ nhận diện hơn.
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_img = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

#Giúp ảnh rõ ràng hơn bằng cách tăng độ sắc nét cạnh.
def apply_unsharp_mask(image, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened
#Làm sáng các vùng mờ nhạt, điều chỉnh range pixel:
def linear_contrast_stretch(image):
    min_val = np.percentile(image, 2)
    max_val = np.percentile(image, 98)
    stretched = np.clip((image - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
    return stretched

#segment kmeans
def segment_by_kmeans(image, k=2):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))

    # Lấy mask của cluster tối nhất (vì tổn thương hay mảng da bất thường thường có màu sẫm)
    darkest_cluster_idx = np.argmin(np.sum(center, axis=1))  # Tổng RGB thấp nhất
    mask = (label.flatten() == darkest_cluster_idx).astype(np.uint8)
    mask = mask.reshape((image.shape[0], image.shape[1]))

    return segmented_image, mask
#remove hairs
def remove_hairs(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply black-hat filtering to detect dark lines (hair)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create a mask of hair
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint to remove the hair from image
    inpainted = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    return inpainted

def segment_by_otsu(gray_image):
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask

def segment_with_otsu_and_kmeans(image, k=2):
    # Bước 1: Áp dụng Otsu Thresholding
    gray = color.rgb2gray(image)
    gray_image = img_as_ubyte(gray)
    
    # Áp dụng Otsu Thresholding để phân đoạn ảnh
    otsu_mask = segment_by_otsu(gray_image)
    
    # Bước 2: Phân đoạn ảnh với KMeans
    segmented_image, kmeans_mask = segment_by_kmeans(image, k)
    
    # Bước 3: Kết hợp kết quả Otsu và KMeans
    # Giữ lại vùng có giá trị lớn hơn ngưỡng Otsu và những vùng phân đoạn qua KMeans
    combined_mask = otsu_mask & kmeans_mask  # Chỉ giữ lại các vùng mà cả Otsu và KMeans đều xác nhận
    
    # Áp dụng mask để giữ lại các vùng quan trọng
    masked_image = image.copy()
    masked_image[combined_mask == 0] = 0  # Giữ lại những vùng không phải là nền
    
    return masked_image, otsu_mask, kmeans_mask, combined_mask

def preprocess_fundus_image(image):
    sharpened = apply_unsharp_mask(image)
    
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    clahe_img = apply_clahe(denoised)

    return clahe_img

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
    
def mean_feature(matrix):
    return np.mean(matrix)

def variance_feature(matrix):
    return np.var(matrix)

def sd_feature(matrix):
    return np.std(matrix)
    
def rms_feature(matrix):
    return np.sqrt(np.mean(np.square(matrix)))

list_of_images = []
os.makedirs(os.path.join(base_path,"data/Dataset/Fusion_feature"), exist_ok=True)

for i in range(1,4):
    path_of_images =os.path.join( base_path,f"data/Image/imgs_part_{i}")
    images = os.listdir(path_of_images)
    list_of_images.extend([os.path.join(path_of_images, img) for img in images])


if os.path.exists(os.path.join(base_path,"data/Dataset/Fusion_feature/images_ft.csv")):
    os.remove(os.path.join(base_path,"data/Dataset/Fusion_feature/images_ft.csv"))
if not os.path.exists(os.path.join(base_path,"data/Dataset/Fusion_feature/images_ft.csv")):
    with open(os.path.join(base_path,"data/Dataset/Fusion_feature/images_ft.csv"), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "img_id",
            "Contrast Feature",
            "Dissimilarity Feature",
            "Homogeneity Feature",
            "Energy Feature",
            "Correlation Feature",
            "ASM Feature",
        ])
for image in list_of_images:
    img = cv2.imread(image)
    img = cv2.resize(img, (256, 256))
    img = preprocess_fundus_image(img)
    segmented_img, otsu_mask, kmeans_mask, combined_mask = segment_with_otsu_and_kmeans(img, k=3)
    
    time_segment_image = time.time()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bước 5: Áp dụng mask Otsu để giữ lại vùng tổn thương
    masked_img = cv2.bitwise_and(gray_img, gray_img, mask=otsu_mask)

    # Bước 6: Chuyển ảnh masked về dạng 8-bit nếu cần
    masked_img_ubyte = img_as_ubyte(masked_img)

    # Bước 7: Phân bin ảnh thành 16 mức xám

    # Bước 8: Tính GLCM
    matrix_coocurrence = graycomatrix(
        masked_img_ubyte,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        normed=True,
        symmetric=True,
    )
    masked_inds = img[otsu_mask == 0]
    CF = contrast_feature(matrix_coocurrence)
    DF= dissimilarity_feature(matrix_coocurrence)
    HF = homogeneity_feature(matrix_coocurrence)
    EF = energy_feature(matrix_coocurrence)
    COR = correlation_feature(matrix_coocurrence)
    ASM = asm_feature(matrix_coocurrence)
    
    with open(os.path.join(base_path,"data/Dataset/Fusion_feature/images_ft.csv"), mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            os.path.basename(image),
            CF,
            DF,
            HF,
            EF,
            COR,
            ASM
        ])
    
    del img, gray_img, masked_img, masked_img_ubyte, matrix_coocurrence,CF,DF,HF,EF,COR,ASM


df = pd.read_csv(os.path.join(base_path,"data/Dataset/Fusion_feature/images_ft.csv"))
df = pd.DataFrame(df)
for col in df.columns[1:]:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
print(df)


dfMetaData = pd.read_csv(os.path.join(base_path,"data/Dataset/metadata.csv"))
dfMetaData = dfMetaData[["age", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed","img_id", "diagnostic"]]
dfMerge = dfMetaData.merge(df, how='inner', on='img_id')
print(dfMerge)
columns = [col for col in dfMerge.columns if col != 'diagnostic']
dfMerge = dfMerge[columns + ['diagnostic']]
mapping = {'BCC': 1, 'SCC': 2, 'ACK': 3, 'SEK' : 4, 'NEV': 5, 'MEL':6}
dfMerge['diagnostic']=dfMerge['diagnostic'].replace(mapping)
dfMerge = dfMerge.drop(['img_id'],axis=1)
# dfMerge = dfMerge.drop(columns=['Unnamed: 0'], errors='ignore')
columns_to_normalize = ["Variance Feature", "Standard Deviation Feature", "RMS Feature","Mean Feature"]

for col in columns_to_normalize:
    if col in dfMerge.columns:
        dfMerge[col] = (dfMerge[col] - dfMerge[col].min()) / (dfMerge[col].max() - dfMerge[col].min())
gender_mapping = {
    'FEMALE': 0,
    'MALE': 1
}

# Mapping cho trạng thái TRUE/FALSE/UNK
boolean_mapping = {
    'FALSE': 0,
    'TRUE': 1,
    'UNK': 2
}

# Mapping cho các quốc gia (Châu Âu, Châu Mỹ, Châu Á)
country_mapping = {
    # Châu Âu
    'POMERANIA': 1,
    'GERMANY': 2,
    'NETHERLANDS': 3,
    'ITALY': 4,
    'POLAND': 5,
    'PORTUGAL': 6,
    'CZECH': 7,
    'NORWAY': 8,
    'SPAIN': 9,
    'AUSTRIA': 10,
    'FRANCE': 11,
    # Châu Mỹ
    'BRAZIL': 12,
    # Châu Á
    'ISRAEL': 13
}

# Mapping cho các vùng trên cơ thể
body_area_mapping = {
    # Vùng đầu
    'FACE': 1,
    'SCALP': 2,
    'NOSE': 3,
    'EAR': 4,
    'LIP': 5,
    
    # Vùng cổ và thân trên
    'NECK': 6,
    'CHEST': 7,
    'BACK': 8,
    'ABDOMEN': 9,
    
    # Vùng tay
    'ARM': 10,
    'FOREARM': 11,
    'HAND': 12,
    
    # Vùng chân
    'THIGH': 13,
    'FOOT': 14
}


dfMerge['region'] = dfMerge['region'].replace(body_area_mapping)

for col in ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation', 'biopsed']:
    dfMerge[col] = dfMerge[col].replace(boolean_mapping)
boolean_mapping = {
    False: 0,
    True: 1
}

dfMerge['biopsed'] = dfMerge['biopsed'].replace(boolean_mapping)

# corr_with_label = dfMerge.corr()['diagnostic'].abs().sort_values(ascending=False)
# selected_features = corr_with_label[corr_with_label > 0.02].index.tolist()

# if 'diagnostic' not in selected_features:
#     selected_features.append('diagnostic')
# df_selected = dfMerge[selected_features]

dfMerge.to_csv(os.path.join(base_path,"data/Dataset/FusionFeatureRemoveMissing.csv"), index=False)
end = time.time()
print(f'Process time: ${start-end}s')


import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = dfMerge.corr()
dfMerge = pd.DataFrame(corr_matrix)
dfMerge.to_csv(os.path.join(base_path,"data/Dataset/Fusion_remove_missing/correlation_matrix.csv"), index=False)

