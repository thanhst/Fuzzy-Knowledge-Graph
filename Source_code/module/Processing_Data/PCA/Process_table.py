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
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

base_path = Path(__file__).resolve().parents[2]

def process(file_path="data/Dataset_diabetic/labels_brset.csv",folder="Table Feature",n_components = 10): 
    os.makedirs(os.path.join(base_path,f'data/Dataset_diabetic/{folder}'), exist_ok=True)
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

    pca = PCA(n_components=n_components)
    

    time_selection_data = time.time()
    X = dfData_scaled.iloc[:, :-1]
    y = dfData_scaled.iloc[:, -1]
    
    X_pca = pca.fit_transform(X)
    dfPCA = pd.DataFrame(np.hstack((X_pca, y.values.reshape(-1, 1))))
    
    X_pca = dfPCA.iloc[:, :-1]
    y_pca = dfPCA.iloc[:, -1]
    df_merged = pd.concat([X_pca, y_pca], axis=1)

    df_merged.insert(0, index_name, index_values)
    pd.DataFrame(df_merged).to_csv(os.path.join(base_path,f'data/Dataset_diabetic/{folder}/table_fts.csv'), index=False)

    print(f'Time process normalization data: {total_time_normalization_data_table}')
    

    end_time_selection_data = time.time()

    total_time_selected = end_time_selection_data - time_selection_data

    print(f'Time selected data: {total_time_selected}')

    total_time_process_data = total_time_preprocess_data_table + total_time_selected + total_time_normalization_data_table

    print(f'Total time for process table data: {total_time_process_data}')