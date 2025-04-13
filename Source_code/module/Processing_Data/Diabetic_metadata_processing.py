import pandas as pd
import numpy as np
import os

from pathlib import Path
base_path = Path(__file__).resolve().parents[2]

dfMerge = pd.read_csv(os.path.join(base_path,"data/Dataset_diabetic/labels_brset.csv"))
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.utils import shuffle

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

dfMerge = dfMerge[columns_to_keep]
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
border_smote = BorderlineSMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
X_resampled, y_resampled = border_smote.fit_resample(X, y)
# Gộp lại thành DataFrame mới
dfBalanced = pd.concat([
    pd.DataFrame(X_resampled, columns=X.columns),
    pd.Series(y_resampled, name='diabetic_retinopathy')
], axis=1)

# Shuffle lại dữ liệu để tránh bias
dfBalanced = shuffle(dfBalanced, random_state=42)
dfMerge = pd.DataFrame(dfBalanced)

os.makedirs(os.path.join(base_path,"data/Dataset_diabetic/Metadata_feature"), exist_ok=True)
dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/Metadata_feature/data_process.csv"),index=False)

import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = dfMerge.corr()
dfMerge = pd.DataFrame(corr_matrix)
dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/Metadata_feature/correlation_matrix.csv"), index=False)
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Ma trận tương quan của các đặc trưng table")
# plt.show()