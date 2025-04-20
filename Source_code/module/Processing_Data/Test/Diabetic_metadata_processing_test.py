import pandas as pd
import numpy as np
import os

from pathlib import Path
base_path = Path(__file__).resolve().parents[3]
import time

from imblearn.over_sampling import BorderlineSMOTE
from sklearn.utils import shuffle
output_dir = os.path.join(base_path, "data/Dataset_diabetic/Metadata_feature_test")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)
dfMerge = pd.read_csv(os.path.join(base_path,"data/Dataset_diabetic/test_data.csv"))
start_process_table = time.time()
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
# dfMerge = dfData
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
labels = {
    1:2,
    0:1,
}
dfMerge['diabetic_retinopathy'] = dfMerge['diabetic_retinopathy'].replace(labels)
dfMerge['diabetes_time_y'] = dfMerge['diabetes_time_y'].replace(diabetes_time_y_process)
dfMerge['insuline'] = dfMerge['insuline'].replace(insuline_process)
dfMerge['diabetes'] = dfMerge['diabetes'].replace(diabetes_process)
dfMerge['quality'] = dfMerge['quality'].replace(quality_process)
dfMerge['insuline'] = pd.to_numeric(dfMerge['insuline'], errors='coerce')
dfMerge['diabetes'] = pd.to_numeric(dfMerge['diabetes'], errors='coerce')
dfMerge['diabetes_time_y'] = pd.to_numeric(dfMerge['diabetes_time_y'], errors='coerce')
end_time_normalization = time.time()
total_time_normalization_data_table = end_time_normalization - start_process_table

start_fill_na = time.time()
dfMerge['diabetes_time_y'] = dfMerge['diabetes_time_y'].fillna(dfMerge['diabetes_time_y'].mean())
dfMerge['insuline'] = dfMerge['insuline'].fillna(dfMerge['insuline'].mean())
dfMerge['diabetes'] = dfMerge['diabetes'].fillna(dfMerge['diabetes'].mean())
dfMerge['patient_age'] = dfMerge['patient_age'].fillna(dfMerge['patient_age'].mean())

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
end_missing = time.time()
total_time_preprocess_data_table=  end_missing-start_fill_na

print(f'Time process missing data: {total_time_preprocess_data_table}')
print(f'Time process normalization data: {total_time_normalization_data_table}')

time_selection_data = time.time()
# Lọc đặc trưng bằng corr
corr_with_label = dfBalanced.corr()['diabetic_retinopathy'].abs().sort_values(ascending=False)
selected_features = corr_with_label[corr_with_label > 0.02].index.tolist()


if 'diabetic_retinopathy' not in selected_features:
    selected_features.append('diabetic_retinopathy')
df_selected = dfBalanced[selected_features]
end_time_selection_data = time.time()

total_time_selected = end_time_selection_data - time_selection_data
dfMerge = pd.DataFrame(dfBalanced)
dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/Metadata_feature_test/data_process.csv"),index=False)


import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = dfMerge.corr()
dfMerge = pd.DataFrame(corr_matrix)
dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/Metadata_feature_test/correlation_matrix.csv"), index=False)
print(f'Time selected data: {total_time_selected}')

total_time_process_data = total_time_preprocess_data_table + total_time_selected + total_time_normalization_data_table

print(f'Total time for process table data: {total_time_process_data}')