# import pandas as pd
# import numpy as np
# import os

# from pathlib import Path
# base_path = Path(__file__).resolve().parents[2]
# import time

# from imblearn.over_sampling import BorderlineSMOTE
# from sklearn.utils import shuffle
# output_dir = os.path.join(base_path, "data/Dataset_diabetic/Metadata_feature")

# os.makedirs(output_dir, exist_ok=True)
# dfMerge = pd.read_csv(os.path.join(base_path,"data/Dataset_diabetic/labels_brset.csv"))
# start_process_table = time.time()
# columns_to_keep = [
#     'image_id',
#     'patient_age',
#     'patient_sex',
#     'diabetes_time_y',
#     'insuline',
#     'diabetes',
#     'exam_eye',
#     'optic_disc',
#     'vessels',
#     'macula',
#     'focus',
#     'Illuminaton',
#     'image_field',
#     'quality',
#     'diabetic_retinopathy'
# ]

# dfMerge = dfMerge[columns_to_keep]
# # dfMerge = dfData
# diabetes_time_y_process = {
#     'NA':0
# }

# quality_process = {
#     'Adequate':2,
#     'Inadequate':1
# }
# insuline_process = {
#     'yes':2,
#     'no':1,
# }
# diabetes_process = {
#     'yes':2,
#     'No':1,
# }
# labels = {
#     1:2,
#     0:1,
# }
# dfMerge['diabetic_retinopathy'] = dfMerge['diabetic_retinopathy'].replace(labels)
# dfMerge['diabetes_time_y'] = dfMerge['diabetes_time_y'].replace(diabetes_time_y_process)
# dfMerge['insuline'] = dfMerge['insuline'].replace(insuline_process)
# dfMerge['diabetes'] = dfMerge['diabetes'].replace(diabetes_process)
# dfMerge['quality'] = dfMerge['quality'].replace(quality_process)
# dfMerge['insuline'] = pd.to_numeric(dfMerge['insuline'], errors='coerce')
# dfMerge['diabetes'] = pd.to_numeric(dfMerge['diabetes'], errors='coerce')
# dfMerge['diabetes_time_y'] = pd.to_numeric(dfMerge['diabetes_time_y'], errors='coerce')
# end_time_normalization = time.time()
# total_time_normalization_data_table = end_time_normalization - start_process_table

# start_fill_na = time.time()
# dfMerge['diabetes_time_y'] = dfMerge['diabetes_time_y'].fillna(dfMerge['diabetes_time_y'].mean())
# dfMerge['insuline'] = dfMerge['insuline'].fillna(dfMerge['insuline'].mean())
# dfMerge['diabetes'] = dfMerge['diabetes'].fillna(dfMerge['diabetes'].mean())
# dfMerge['patient_age'] = dfMerge['patient_age'].fillna(dfMerge['patient_age'].mean())

# dfMerge = dfMerge.drop(['image_id'],axis=1)
# dfMerge = dfMerge[[col for col in dfMerge.columns if col != 'diabetic_retinopathy'] + ['diabetic_retinopathy']]

# X = dfMerge.drop('diabetic_retinopathy', axis=1)
# y = dfMerge['diabetic_retinopathy']
# # Áp dụng BorderlineSMOTE
# # Xử lý dữ liệu mất cân bằng
# border_smote = BorderlineSMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
# X_resampled, y_resampled = border_smote.fit_resample(X, y)

# # Gộp lại thành DataFrame mới
# dfBalanced = pd.concat([
#     pd.DataFrame(X_resampled, columns=X.columns),
#     pd.Series(y_resampled, name='diabetic_retinopathy')
# ], axis=1)

# # Shuffle lại dữ liệu để tránh bias
# dfBalanced = shuffle(dfBalanced, random_state=42)
# end_missing = time.time()
# total_time_preprocess_data_table=  end_missing-start_fill_na

# print(f'Time process missing data: {total_time_preprocess_data_table}')
# print(f'Time process normalization data: {total_time_normalization_data_table}')

# time_selection_data = time.time()
# # Lọc đặc trưng bằng corr
# corr_with_label = dfBalanced.corr()['diabetic_retinopathy'].abs().sort_values(ascending=False)
# selected_features = corr_with_label[corr_with_label > 0.02].index.tolist()


# if 'diabetic_retinopathy' not in selected_features:
#     selected_features.append('diabetic_retinopathy')
# df_selected = dfBalanced[selected_features]
# end_time_selection_data = time.time()

# total_time_selected = end_time_selection_data - time_selection_data
# dfMerge = pd.DataFrame(dfBalanced)
# dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/Metadata_feature/data_process.csv"),index=False)


# import matplotlib.pyplot as plt
# import seaborn as sns
# corr_matrix = dfMerge.corr()
# dfMerge = pd.DataFrame(corr_matrix)
# dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/Metadata_feature/correlation_matrix.csv"), index=False)
# print(f'Time selected data: {total_time_selected}')

# total_time_process_data = total_time_preprocess_data_table + total_time_selected + total_time_normalization_data_table

# print(f'Total time for process table data: {total_time_process_data}')
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import os
import cv2
import cupy as cp  # Sử dụng cupy thay vì numpy
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.utils import shuffle
from pathlib import Path
import time
import gc
import csv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

base_path = Path(__file__).resolve().parents[2]
def process_table(file_path,folder_save):
    os.makedirs(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}'), exist_ok=True)
    dfData = pd.read_csv(os.path.join(base_path,file_path))
    index = dfData.iloc[:, 0]  # Lấy cột đầu tiên
    index_name = dfData.columns[0]
    index_values = index.values
    dfData = dfData.iloc[:, 1:]
    start_time_process_zero = time.time()
    for col in dfData.columns:
        le = LabelEncoder()
        dfData[col] = dfData[col].replace(['', ' ', None], pd.NA)
        dfData[col] = dfData[col].fillna('NA')
        dfData[col] = le.fit_transform(dfData[col].astype(str))  # ép thành chuỗi nếu có NaN
    end_time_process_zero = time.time()
    total_time_preprocess_data_table = end_time_process_zero - start_time_process_zero
    # 2. Chuẩn hóa tất cả các cột về [0, 1]
    start_norm = time.time()
    scaler = MinMaxScaler()
    dfData_scaled = pd.DataFrame(scaler.fit_transform(dfData), columns=dfData.columns)
    end_norm = time.time()
    total_time_normalization_data_table = end_norm - start_norm

    print(f'Time process normalization data: {total_time_normalization_data_table}')
    
    time_selection_data = time.time()

    df_Ftab = dfData_scaled
    df_Ftab.insert(0, index_name, index_values)
    df_Ftab.to_csv(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/table_fts.csv'), index=False)
    
    end_time_selection_data = time.time()

    total_time_selected = end_time_selection_data - time_selection_data

    print(f'Time selected data: {total_time_selected}')

    total_time_process_data = total_time_preprocess_data_table + total_time_selected + total_time_normalization_data_table

    print(f'Total time for process table data: {total_time_process_data}')

def processing_ft_selection(file_path_img,file_path_table, folder_save,k):
    os.makedirs(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}'), exist_ok=True)
    process_table(file_path=file_path_table,folder_save=folder_save)
    df_table = pd.read_csv(f'data/Dataset_diabetic/{folder_save}/table_fts.csv')
    # df_merge = df_image.merge(df_table.iloc[:,:-1],how='inner',on="image_id")
    # df_merge = df_merge.drop(['image_id'],axis=1)
    
    Label = df_table.iloc[:,-1]
    mid = df_table.shape[1] // 2
    FAll = df_table.iloc[:,1:-1]
    Ftab = df_table.iloc[:, 1:mid]
    Fimg = df_table.iloc[:, mid:-1]
    
    from sklearn.preprocessing import StandardScaler
    scaler_img = StandardScaler()
    scaler_tab = StandardScaler()
    Fimg = scaler_img.fit_transform(Fimg.values)
    Ftab = scaler_tab.fit_transform(Ftab.values)
    FAll = scaler_tab.fit_transform(FAll.values)
    # Áp dụng wrapper
    # from module.Processing_Data.Fusion_function import wrapper_multimodal_feature_selection
    # Ffused_df = pd.DataFrame(wrapper_multimodal_feature_selection.wrapper_multimodal_selection(Fimg=Fimg,Ftab=Ftab,target=Label.values.ravel(),max_img=max_img,max_tab=max_tab,min_img=2,min_tab=2))
    
    # FT selection
    from module.Processing_Data.Fusion_function import feature_selection
    Ffused_df = pd.DataFrame(feature_selection.select_features(FAll,Label,k = k))
    Ffused_df = pd.concat([Ffused_df, Label.reset_index(drop=True)], axis=1)
    
    #Filter
    # from module.Processing_Data.Fusion_function import filter_multimodal_selection
    # Ffused_df = pd.DataFrame(filter_multimodal_selection.filter_multimodal_selection(Fimg=Fimg,Ftab=Ftab,target=y,k_img=5,k_tab=4))
    # Ffused_df = pd.concat([Ffused_df, Label.reset_index(drop=True)], axis=1)
    
    
    # from module.Processing_Data.Fusion_function import hadamard_selection
    # Ffused_df = pd.DataFrame(hadamard_selection.hadamard_fusion(Fimg=Fimg,Ftab=Ftab,common_dim=3))
    # Ffused_df = pd.concat([Ffused_df, Label.reset_index(drop=True)], axis=1)
    
    # from module.Processing_Data.Fusion_function import tensor_selection
    # Ffused_df = pd.DataFrame(tensor_selection.tensor_fusion(Fimg=Fimg,Ftab=Ftab,rank=9))
    # Ffused_df = pd.concat([Ffused_df, Label.reset_index(drop=True)], axis=1)
    
    X = Ffused_df.iloc[:, :-1]
    y = Ffused_df.iloc[:, -1]
    X.columns = X.columns.astype(str)
    # Áp dụng BorderlineSMOTE
    # Xử lý dữ liệu mất cân bằng
    border_smote = BorderlineSMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
    X_resampled, y_resampled = border_smote.fit_resample(X, y)

    # Gộp lại thành DataFrame mới
    dfBalanced = pd.concat([
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled, name='diabetic_retinopathy')
    ], axis=1)
    dfBalanced.to_csv(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/data_process.csv'),index=False)
    
def processing_wrapper(file_path_img,file_path_table, folder_save,max_img,max_tab):
    os.makedirs(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}'), exist_ok=True)
    process_table(file_path=file_path_table,folder_save=folder_save)
    df_table = pd.read_csv(f'data/Dataset_diabetic/{folder_save}/table_fts.csv')
    # df_merge = df_image.merge(df_table.iloc[:,:-1],how='inner',on="image_id")
    # df_merge = df_merge.drop(['image_id'],axis=1)
    
    Label = df_table.iloc[:,-1]
    mid = df_table.shape[1] // 2
    FAll = df_table.iloc[:,1:-1]
    Ftab = df_table.iloc[:, 1:mid]
    Fimg = df_table.iloc[:, mid:-1]
    
    from sklearn.preprocessing import StandardScaler
    scaler_img = StandardScaler()
    scaler_tab = StandardScaler()
    Fimg = scaler_img.fit_transform(Fimg.values)
    Ftab = scaler_tab.fit_transform(Ftab.values)
    
    # Áp dụng wrapper
    from module.Processing_Data.Fusion_function import wrapper_multimodal_feature_selection
    Ffused_df = pd.DataFrame(wrapper_multimodal_feature_selection.wrapper_multimodal_selection(Fimg=Fimg,Ftab=Ftab,target=Label.values.ravel(),max_img=max_img,max_tab=max_tab,min_img=2,min_tab=2))
    
    # FT selection
    # from module.Processing_Data.Fusion_function import feature_selection
    # Ffused_df = pd.DataFrame(feature_selection.select_features(FAll,Label,k = k))
    # Ffused_df = pd.concat([Ffused_df, Label.reset_index(drop=True)], axis=1)
    
    #Filter
    # from module.Processing_Data.Fusion_function import filter_multimodal_selection
    # Ffused_df = pd.DataFrame(filter_multimodal_selection.filter_multimodal_selection(Fimg=Fimg,Ftab=Ftab,target=y,k_img=5,k_tab=4))
    # Ffused_df = pd.concat([Ffused_df, Label.reset_index(drop=True)], axis=1)
    
    
    # from module.Processing_Data.Fusion_function import hadamard_selection
    # Ffused_df = pd.DataFrame(hadamard_selection.hadamard_fusion(Fimg=Fimg,Ftab=Ftab,common_dim=3))
    # Ffused_df = pd.concat([Ffused_df, Label.reset_index(drop=True)], axis=1)
    
    # from module.Processing_Data.Fusion_function import tensor_selection
    # Ffused_df = pd.DataFrame(tensor_selection.tensor_fusion(Fimg=Fimg,Ftab=Ftab,rank=9))
    # Ffused_df = pd.concat([Ffused_df, Label.reset_index(drop=True)], axis=1)
    
    X = Ffused_df.iloc[:, :-1]
    y = Ffused_df.iloc[:, -1]
    X.columns = X.columns.astype(str)
    # Áp dụng BorderlineSMOTE
    # Xử lý dữ liệu mất cân bằng
    border_smote = BorderlineSMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
    X_resampled, y_resampled = border_smote.fit_resample(X, y)

    # Gộp lại thành DataFrame mới
    dfBalanced = pd.concat([
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled, name='diabetic_retinopathy')
    ], axis=1)
    dfBalanced.to_csv(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/data_process.csv'),index=False)