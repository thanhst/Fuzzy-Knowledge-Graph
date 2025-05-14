import os
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
import csv
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
def preprocessing_data(file_path,folder_save):
    list_of_npz = []
    path_of_npz = os.path.join(base_path,file_path)
    
    npzs = os.listdir(path_of_npz)
    list_of_npz.extend([
    os.path.join(path_of_npz, file)
    for file in npzs
    if file.endswith('.npz')
    ])
    # if os.path.exists(os.path.join(base_path,f'data/{folder_save}/images_ft.csv')):
    #     os.remove(os.path.join(base_path,f'data/{folder_save}/images_ft.csv'))
    if not os.path.exists(os.path.join(base_path,f'data/{folder_save}/table_ft.csv')):
        with open(os.path.join(base_path,f'data/{folder_save}/table_ft.csv'), mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_id",
                "race",
                "male",
                "hispanic",
                "maritalstatus",
                "language",
                "dr_subtype",
                "dr_class",
            ])

    for npz in list_of_npz:
        data= np.load(npz)
        race = data['race']
        male = data['male']
        hispanic =data['hispanic']
        maritalstatus = data['maritalstatus']
        language = data['language']
        dr_class= data['dr_class']
        dr_subtype = data['dr_class']
        
        with open(os.path.join(base_path,f'data/{folder_save}/table_ft.csv'), mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                os.path.splitext(os.path.basename(npz))[0],
                race,
                male,
                hispanic,
                maritalstatus,
                language,
                dr_subtype,
                dr_class,
            ])

        df = pd.read_csv(os.path.join(base_path,f'data/{folder_save}/table_ft.csv'))
        
def process_corr_advanced(file_path, folder_save, label_column=None, is_classification=True):
    df = pd.read_csv(file_path)

    # Xác định nhãn
    if label_column is None:
        label_column = df.columns[-1]
    
    label = df[label_column]
    features = df.drop(columns=[label_column])

    # Encode label nếu cần
    if label.dtype == 'object' or label.dtype.name == 'category':
        label = LabelEncoder().fit_transform(label)

    # Pearson
    pearson_corr = features.corrwith(pd.Series(label), method='pearson')

    # Spearman (monotonic, tuyến tính phi chuẩn)
    spearman_corr = features.corrwith(pd.Series(label), method='spearman')

    # Mutual Information
    if is_classification:
        mi = mutual_info_classif(features, label, discrete_features='auto')
    else:
        mi = mutual_info_regression(features, label, discrete_features='auto')

    # Tổng hợp bảng kết quả
    result_df = pd.DataFrame({
        'Feature': features.columns,
        'Pearson': pearson_corr,
        'Spearman': spearman_corr,
        'Mutual_Info': mi
    }).sort_values(by='Mutual_Info', ascending=False)

    # Lưu bảng ra file CSV
    result_df.to_csv(os.path.join(base_path,f"data\{folder_save}/feature_correlation_summary.csv"), index=False)

    # Vẽ heatmap các hệ số tương quan
    sns.heatmap(result_df[['Pearson', 'Spearman', 'Mutual_Info']].set_index(result_df['Feature']), annot=True, cmap='YlGnBu')
    plt.title('Tương quan & Thông tin lẫn nhau với nhãn')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path,f"data\{folder_save}/feature_correlation_advanced.png"))
    plt.close()

    return result_df