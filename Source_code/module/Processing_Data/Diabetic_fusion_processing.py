import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import os
import cv2
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.utils import shuffle
from pathlib import Path
base_path = Path(__file__).resolve().parents[2]

#Làm rõ vùng tối/sáng, giúp mạch máu và tổn thương dễ nhận diện hơn.
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

def segment_by_otsu(gray_image):
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask

def preprocess_fundus_image(image):

    # Làm nét
    sharpened = apply_unsharp_mask(image)

    # Cải thiện độ tương phản bằng CLAHE
    clahe_img = apply_clahe(sharpened)
    # Stretch nhẹ để làm sáng
    final = linear_contrast_stretch(clahe_img)

    _, lesion_mask = segment_by_kmeans(final)
    
    return final,lesion_mask


df = pd.DataFrame(
    columns=[
        "Contrast Feature",
        "Dissimilarity Feature",
        "Homogeneity Feature",
        "Energy Feature",
        "Correlation Feature",
        "ASM Feature",
        "Mean Feature",
        "Variance Feature",
        "Standard Deviation Feature",
        "RMS Feature"
    ]
)
matrix1 = []
list_of_images = []
name_of_images = []
path_of_images = os.path.join(base_path,"data/Image/fundus_photo/")
images = os.listdir(path_of_images)
name_of_images.extend([os.path.splitext(f)[0] for f in os.listdir(path_of_images)])
list_of_images.extend([os.path.join(path_of_images, img) for img in images])
    
for image in list_of_images:
    img = cv2.imread(image)
    img, mask = preprocess_fundus_image(img)
    gray = color.rgb2gray(img)
    image = img_as_ubyte(gray)
    gray_masked = gray.copy()
    gray_masked[mask == 0] = 0
    bins = np.array(
        [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
    )  # 16-bit
    inds = np.digitize(
        gray_masked, bins
    )

    max_value = inds.max() + 1
    matrix_coocurrence = graycomatrix(
        inds,
        [1],
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=max_value,
        normed=False,
        symmetric=False,
    )
    matrix1.append(matrix_coocurrence)
CF =[]
DF =[]
HF =[]
EF =[]
COR = []
ASM = []
MF, VF, SD ,RMS = [], [], [],[]

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
    
for matrix in matrix1:
    CF.append(contrast_feature(matrix))
    DF.append(dissimilarity_feature(matrix))
    HF.append(homogeneity_feature(matrix))
    EF.append(energy_feature(matrix))
    COR.append(correlation_feature(matrix))
    ASM.append(asm_feature(matrix))
    MF.append(mean_feature(matrix))
    VF.append(variance_feature(matrix))
    SD.append(sd_feature(matrix))
    RMS.append(rms_feature(matrix))
df["Contrast Feature"] = CF
df["Dissimilarity Feature"] = DF
df["Homogeneity Feature"] = HF
df["Energy Feature"] = EF
df["Correlation Feature"] = COR
df["ASM Feature"] = ASM
df["Mean Feature"] = MF
df["Variance Feature"] = VF
df["Standard Deviation Feature"] = SD
df["RMS Feature"] = RMS

df.index = name_of_images
df.index.name = "image_id"

dfData = pd.read_csv(os.path.join(base_path,"data/Dataset_diabetic/labels_brset.csv"))

columns_to_keep = [
    'image_id',
    'patient_age',
    'patient_sex',
    'diabetes_time_y',
    'insuline',
    'diabetes',
    'exam_eye',
    'optic_disc',
    'vessels',
    'macula',
    'focus',
    'Illuminaton',
    'image_field',
    'quality',
    'diabetic_retinopathy'
]

dfData = dfData[columns_to_keep]
# dfMerge = dfData
dfMerge = dfData.merge(df,how = 'inner', on= 'image_id')
columns_to_normalize = ["Variance Feature", "Standard Deviation Feature", "RMS Feature","Mean Feature"]

for col in columns_to_normalize:
    if col in dfMerge.columns:
        dfMerge[col] = (dfMerge[col] - dfMerge[col].min()) / (dfMerge[col].max() - dfMerge[col].min())

diabetes_time_y_process = {
    'NA':0
}

quality_process = {
    'Adequate':2,
    'Inadequate':1
}
insuline_process = {
    'yes':2,
    'no':1,
}
diabetes_process = {
    'yes':2,
    'No':1,
}

dfMerge['diabetes_time_y'] = dfMerge['diabetes_time_y'].replace(diabetes_time_y_process)
dfMerge['insuline'] = dfMerge['insuline'].replace(insuline_process)
dfMerge['insuline'] = pd.to_numeric(dfMerge['insuline'], errors='coerce')
dfMerge['diabetes'] = dfMerge['diabetes'].replace(diabetes_process)
dfMerge['quality'] = dfMerge['quality'].replace(quality_process)
dfMerge['diabetes'] = pd.to_numeric(dfMerge['diabetes'], errors='coerce')
dfMerge['diabetes_time_y'] = pd.to_numeric(dfMerge['diabetes_time_y'], errors='coerce')
dfMerge['diabetes_time_y'] = dfMerge['diabetes_time_y'].fillna(dfMerge['diabetes_time_y'].mean())
dfMerge['insuline'] = dfMerge['insuline'].fillna(dfMerge['insuline'].mean())
dfMerge['diabetes'] = dfMerge['diabetes'].fillna(dfMerge['diabetes'].mean())
dfMerge['patient_age'] = dfMerge['patient_age'].fillna(dfMerge['patient_age'].mean())

labels = {
    1:2,
    0:1,
}
dfMerge['diabetic_retinopathy'] = dfMerge['diabetic_retinopathy'].replace(labels)


dfMerge = dfMerge.drop(['image_id'],axis=1)
dfMerge = dfMerge[[col for col in dfMerge.columns if col != 'diabetic_retinopathy'] + ['diabetic_retinopathy']]

X = dfMerge.drop('diabetic_retinopathy', axis=1)
y = dfMerge['diabetic_retinopathy']
# Áp dụng BorderlineSMOTE
# Xử lý dữ liệu mất cân bằng
border_smote = BorderlineSMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
X_resampled, y_resampled = border_smote.fit_resample(X, y)
# Gộp lại thành DataFrame mới
dfBalanced = pd.concat([
    pd.DataFrame(X_resampled, columns=X.columns),
    pd.Series(y_resampled, name='diabetic_retinopathy')
], axis=1)

# Shuffle lại dữ liệu để tránh bias
dfBalanced = shuffle(dfBalanced, random_state=42)

#Lọc đặc trưng bằng corr
corr_with_label = dfBalanced.corr()['diabetic_retinopathy'].abs().sort_values(ascending=False)
selected_features = corr_with_label[corr_with_label > 0.02].index.tolist()

if 'diabetic_retinopathy' not in selected_features:
    selected_features.append('diabetic_retinopathy')
df_selected = dfBalanced[selected_features]

dfMerge = pd.DataFrame(df_selected)
os.makedirs(os.path.join(base_path,"data/Dataset_diabetic/Fusion_feature"), exist_ok=True)
dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/Fusion_feature/data_process.csv"),index=False)

import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = dfMerge.corr()
dfMerge = pd.DataFrame(corr_matrix)
dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/Fusion_feature/correlation_matrix.csv"), index=False)
