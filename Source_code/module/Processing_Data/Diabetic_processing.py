import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import os
import cv2
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
path_of_images = os.path.join(os.path.dirname(base_path),"ImageData/fundus_photo/")
images = os.listdir(path_of_images)
name_of_images.extend([os.path.splitext(f)[0] for f in os.listdir(path_of_images)])
list_of_images.extend([os.path.join(path_of_images, img) for img in images])
    
for image in list_of_images:
    img = cv2.imread(image)
    gray = color.rgb2gray(img)
    image = img_as_ubyte(gray)

    bins = np.array(
        [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
    )
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



dfMerge = dfMerge.drop(['image_id'],axis=1)
dfMerge = dfMerge[[col for col in dfMerge.columns if col != 'diabetic_retinopathy'] + ['diabetic_retinopathy']]
dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/data_process.csv"),index=False)

import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = dfMerge.corr()
dfMerge = pd.DataFrame(corr_matrix)
dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/correlation_matrix.csv"), index=False)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Ma trận tương quan của các đặc trưng ảnh")
plt.show()