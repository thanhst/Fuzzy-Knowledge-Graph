import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def calculate_correlations(file_path, label_column, is_classification=True):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_path)
    
    # Kiểm tra nếu nhãn không tồn tại
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in dataset")
    
    # Tách nhãn và các thuộc tính
    X = df.drop(columns=[label_column])
    y = df[label_column]
    
    # Xử lý các giá trị thiếu
    X = X.select_dtypes(include=[np.number]).fillna(0)  # Chỉ giữ lại cột số
    
    # Tính hệ số tương quan Spearman
    spearman_corr = {col: spearmanr(X[col], y)[0] for col in X.columns}
    
    # Tính Mutual Information
    if is_classification:
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)
    mutual_info = dict(zip(X.columns, mi_scores))
    
    # Kết quả
    result = pd.DataFrame({
        'Feature': X.columns,
        'Spearman Correlation': [spearman_corr[col] for col in X.columns],
        'Mutual Information': [mutual_info[col] for col in X.columns]
    })
    
    result = result.sort_values(by='Mutual Information', ascending=False)
    result.to_csv("./data/Test/corr.csv")

# Cách sử dụng:
# file_path = "your_file.csv"
# label_column = "your_label_column"
# is_classification = True  # Đặt False nếu bài toán là hồi quy
# result = calculate_correlations(file_path, label_column, is_classification)
# print(result)
