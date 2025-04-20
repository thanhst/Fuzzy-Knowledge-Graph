import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os,sys
from pathlib import Path
import time
start = time.time()
base_path = Path(__file__).resolve().parents[2]

dfMetaData = pd.read_csv(os.path.join(base_path ,"data/Dataset/metadata.csv"))

dfDiagnostic = dfMetaData['diagnostic']

dfMetaData = dfMetaData.drop(['diagnostic'], axis=1)

dfMetaData['diagnostic'] = dfDiagnostic
mapping = {'BCC': 1, 'SCC': 2, 'ACK': 3, 'SEK' : 4, 'NEV': 5, 'MEL':6}
dfMetaData['diagnostic']=dfMetaData['diagnostic'].replace(mapping)

dfMetaData = dfMetaData[["age", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed", "diagnostic"]]

# Mapping cho giới tính
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

dfMetaData['region'] = dfMetaData['region'].replace(body_area_mapping)

for col in ['itch', 'grew', 'hurt', 'changed', 'bleed','biopsed', 'elevation']:
    dfMetaData[col] = dfMetaData[col].replace(boolean_mapping)
    
boolean_mapping = {
    False: 0,
    True: 1
}
dfMetaData['biopsed'] = dfMetaData['biopsed'].replace(boolean_mapping)


corr_with_label = dfMetaData.corr()['diagnostic'].abs().sort_values(ascending=False)
selected_features = corr_with_label[corr_with_label > 0.02].index.tolist()

if 'diagnostic' not in selected_features:
    selected_features.append('diagnostic')
df_selected = dfMetaData[selected_features]

dfMetaData.to_csv(os.path.join(base_path,"data/Dataset/OnlyTableFeatureRemoveMissing.csv"), index=False)
end = time.time()
print(f'Process time: ${start-end}s')


dfMetaData_numeric = dfMetaData.select_dtypes(include=[np.number])



corr_matrix = dfMetaData_numeric.corr()
df = pd.DataFrame(corr_matrix)
df.to_csv(os.path.join(base_path,"data/Dataset/OnlyMetadataRemove30%/correlation_matrix.csv"), index=False)

