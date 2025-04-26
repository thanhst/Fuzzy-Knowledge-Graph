import sys
import os

# Lấy đường dẫn tuyệt đối tới thư mục gốc của project (ở đây là Source_code)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # lên 2 cấp

if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from module.FIS.FIS import FIS
from module.FKG.FKG_general import FKG
from module.FKG.FKG_S import FKGS
from sklearn.model_selection import KFold
import numpy as np

print("Diabetic Retinopathy Fusion Feature")

print("__________Running Processing___________")
from module.Processing_Data import Diabetic_fusion_processing_with_pca
Diabetic_fusion_processing_with_pca.processing(file_path="data/Dataset_diabetic/data_process.csv",folder_save="Fusion_feature_PCA",n_components_image=7,n_components_table=10)

print("__________Running FIS___________")
FIS(fileName="Diabetic Retinopathy Feature PCA",
    filePath=os.path.join(project_root,"data\Dataset_diabetic\Fusion_feature_PCA\data_process.csv"),
    cluster=[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,2])
print("--------------------------------")

# print("__________Running FKG___________")
# traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Feature/Rule_List.csv'))
# testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Feature/FRB/TestDataRule.csv'))
# base = [[int(float(x)) for x in row] for row in traindf.values]
# base = pd.DataFrame(base)
# test = [[int(float(x)) for x in row] for row in testdf.values]
# fkg_instance = FKG()
# fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Feature")
# print("--------------------------------")

print("__________Running FKG-S___________")
traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Feature PCA/Rule_List.csv'))
testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Feature PCA/FRB/TestDataRule.csv'))
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKGS()
fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Feature PCA",ran=20,e=0.2)
print("-"*100)

print("__________Running FKG-S with K-fold___________")
traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Feature PCA/Rule_List.csv'))
base = [[int(float(x)) for x in row] for row in traindf.values]
base = np.array(base)
X = base[:, :-1]
y = base[:, -1]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    train_combined = np.hstack((X_train, y_train.reshape(-1, 1)))
    val_combined = np.hstack((X_val, y_val.reshape(-1, 1)))
    train_combined = pd.DataFrame(train_combined)
    fkg_instance = FKGS()
    fkg_instance.FKGS(df = train_combined,testdf=val_combined,Turn=None,Modality="Diabetic Retinopathy Feature PCA",ran=20,e=0.2)
    print("-"*100)


