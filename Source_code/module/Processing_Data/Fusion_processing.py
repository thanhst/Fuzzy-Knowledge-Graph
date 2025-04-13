import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import os
import cv2

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

def preprocess_fundus_image(image):

    # Làm nét
    sharpened = apply_unsharp_mask(image)

    # Cải thiện độ tương phản bằng CLAHE
    clahe_img = apply_clahe(sharpened)
    # Stretch nhẹ để làm sáng
    final = linear_contrast_stretch(clahe_img)

    return final

base_path = os.getcwd()
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
for i in range(1,4):
    path_of_images =os.path.join( base_path,f"data/Image/imgs_part_{i}")
    images = os.listdir(path_of_images)
    name_of_images.extend(images)
    list_of_images.extend([os.path.join(path_of_images, img) for img in images])
    
for image in list_of_images:
    img = cv2.imread(image)
    img = preprocess_fundus_image(img)
    gray = color.rgb2gray(img)
    image = img_as_ubyte(gray) 

    bins = np.array(
        [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
    )  # 16-bit
    inds = np.digitize(
        image, bins
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
df.index.name = "img_id"

dfMetaData = pd.read_csv(os.path.join(base_path,"data/Dataset/metadata.csv"))
dfMetaData = dfMetaData[["age", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed","img_id", "diagnostic"]]
dfMerge = pd.merge(dfMetaData, df, on='img_id', how='inner')
columns = [col for col in dfMerge.columns if col != 'diagnostic']
dfMerge = dfMerge[columns + ['diagnostic']]
mapping = {'BCC': 1, 'SCC': 2, 'ACK': 3, 'SEK' : 4, 'NEV': 5, 'MEL':6}
dfMerge['diagnostic']=dfMerge['diagnostic'].replace(mapping)
dfMerge = dfMerge.drop(['img_id'],axis=1)
dfMerge = dfMerge.drop(columns=['Unnamed: 0'], errors='ignore')
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

dfMerge['biopsed'] = dfMetaData['biopsed'].replace(boolean_mapping)
dfMerge.to_csv(os.path.join(base_path,"data/Dataset/FusionFeatureRemoveMissing.csv"), index=False)



import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = dfMerge.corr()
dfMerge = pd.DataFrame(corr_matrix)
dfMerge.to_csv(os.path.join(base_path,"data/Dataset/Fusion_remove_missing/correlation_matrix.csv"), index=False)
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Ma trận tương quan của các đặc trưng ảnh")
# plt.show()
