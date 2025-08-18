import sys
import os
import numpy as np
# Lấy đường dẫn tuyệt đối tới thư mục gốc của project (ở đây là Source_code)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # lên 2 cấp

if project_root not in sys.path:
    sys.path.append(project_root)
    
from module.FIS.FIS import FIS
from module.FKG.FKG_S_weight_backward import FKG
from module.FKG.FKG_S import FKGS

import pandas as pd

print("Diabetic Retinopathy Table Feature")
epoch = 5
print("__________Running Processing___________")
# from module.Processing_Data import Diabetic_metadata_processing
# # Diabetic_metadata_processing.processing_wrapper(file_path_table="data/Dataset_diabetic/data_process.csv",file_path_img="data/Image/fundus_photo",folder_save="Metadata_feature",max_img=5,max_tab=4)
# Diabetic_metadata_processing.processing_ft_selection(file_path_table="data/Dataset_diabetic/data_process.csv",file_path_img="data/Image/fundus_photo",folder_save="Metadata_feature",k=13)
fkg_instance = FKG(weight=[1]*16, bias=0.0001,learning_rate=0.01)
for i in range(epoch):
    print("Epoch: ", i+1)
    data = pd.read_csv(os.path.join(project_root,'data\Dataset_diabetic\Fusion_feature_FT_selection\data_process_duy_hoang.csv'))
    X_train = data.iloc[:, :-1].values
    X_train = X_train * np.array(fkg_instance.weight) + np.array(fkg_instance.bias)
    X = pd.DataFrame(X_train)
    dataFrame = pd.concat([X, data.iloc[:, -1]], axis=1)
    dataFrame.to_csv(os.path.join(project_root,'data\Dataset_diabetic\Fusion_feature_FT_selection\data_process_duy_hoang.csv'),index=False)
    print("__________Running FIS___________")
    FIS(fileName="Diabetic Retinopathy Hoang Fusion FT Selection",
        filePath=os.path.join(project_root,'data\Dataset_diabetic\Fusion_feature_FT_selection\data_process_duy_hoang.csv'),
        cluster = [3] * 16 + [2])
        # cluster=[5,5,5,5,5,5,5,5,5,2])
    print("--------------------------------")

    print("__________Running FKG___________")
    traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang Fusion FT Selection/FRB/TrainDataRule.csv'))
    testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang Fusion FT Selection/FRB/TestDataRule.csv'))
    base = [[int(float(x)) for x in row] for row in traindf.values]
    base = pd.DataFrame(base)
    test = [[int(float(x)) for x in row] for row in testdf.values]
    fkg_instance.FKG_weight(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Hoang Fusion FT Selection")
    if fkg_instance.loss < 1e-15:
        break
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Hoang Fusion FT Selection")
print("--------------------------------")

e = [0.2,0.3]
r = [15,20]
for i in r:
    for j in e:
        print("__________Running FKG-S___________")
        traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang Fusion FT Selection/FRB/TrainDataRule.csv'))
        testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang Fusion FT Selection/FRB/TestDataRule.csv'))
        base = [[int(float(x)) for x in row] for row in traindf.values]
        base = pd.DataFrame(base)
        test = [[int(float(x)) for x in row] for row in testdf.values]
        fkg_instance = FKGS()
        fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Hoang Fusion FT Selection",ran=i,e=j,folderPath=project_root)
        print("-"*100)

