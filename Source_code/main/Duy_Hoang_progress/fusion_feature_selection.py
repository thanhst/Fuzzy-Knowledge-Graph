import sys
import os
import numpy as np
# Lấy đường dẫn tuyệt đối tới thư mục gốc của project (ở đây là Source_code)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # lên 2 cấp

if project_root not in sys.path:
    sys.path.append(project_root)
    
from module.FIS.FIS_class import FIS
from module.FKG.FKG_S_weight_backward import FKG
from module.FKG.FKG_S import FKGS

import pandas as pd
import time

epoch = 5
print("__________Running Processing___________")
# from module.Processing_Data import Diabetic_metadata_processing
# # Diabetic_metadata_processing.processing_wrapper(file_path_table="data/Dataset_diabetic/data_process.csv",file_path_img="data/Image/fundus_photo",folder_save="Metadata_feature",max_img=5,max_tab=4)
# Diabetic_metadata_processing.processing_ft_selection(file_path_table="data/Dataset_diabetic/data_process.csv",file_path_img="data/Image/fundus_photo",folder_save="Metadata_feature",k=13)
startTime = time.time()
path_file = os.path.join(project_root,'data\Dataset_diabetic\Fusion_feature_wrapper')
fkg_instance = FKG(weight=[1]*16,learning_rate=0.1)
for i in range(epoch):
    print("Epoch: ", i+1)
    file_weight = fkg_instance.Cross_weight(path_file=path_file,file_name = "data_process_duy_hoang",weight=fkg_instance.weight,plus_or_minus="weight")
    print("__________Running FIS___________")
    fis_instance = FIS(fileName="Diabetic Retinopathy Hoang FT Selection",
        filePath=file_weight,
        cluster = [3] * 16 + [2])
        # cluster=[5,5,5,5,5,5,5,5,5,2])
    print("--------------------------------")

    print("__________Running FKG___________")
    traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang FT Selection/FRB/TrainDataRule.csv'))
    testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang FT Selection/FRB/TestDataRule.csv'))
    base = [[int(float(x)) for x in row] for row in traindf.values]
    base = pd.DataFrame(base)
    test = [[int(float(x)) for x in row] for row in testdf.values]
    fkg_instance.Generator_FKG(df = base,testdf=test,Modality="Diabetic Retinopathy Hoang FT Selection")
    
    fis_file = os.path.join(project_root,'data/FIS/input/Diabetic Retinopathy Hoang FT Selection')
    fis_name = 'train_data'
    
    print("__________Start plus weight___________")
    file_plus = fkg_instance.Cross_weight(path_file=fis_file,file_name = fis_name,weight=fkg_instance.weight,h=1e-15,plus_or_minus="plus")
    file_path_to_save = os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang FT Selection/FRB/WeightPlus.csv')
    rules_plus = fis_instance.Generator_rule(file_path_to_gen=file_plus,file_path_to_save=file_path_to_save)
    loss_plus = fkg_instance.FKG_weight(df = rules_plus,testdf=test,Turn=None,Modality="Diabetic Retinopathy Hoang FT Selection")

    print("__________Start minus weight___________")
    file_minus = fkg_instance.Cross_weight(path_file=fis_file,file_name = fis_name,weight=fkg_instance.weight,h=-1e-15,plus_or_minus="minus")
    file_path_to_save = os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang FT Selection/FRB/WeightMinus.csv')
    rules_minus = fis_instance.Generator_rule(file_path_to_gen=file_plus,file_path_to_save=file_path_to_save)
    loss_minus = fkg_instance.FKG_weight(df = rules_minus,testdf=test,Turn=None,Modality="Diabetic Retinopathy Hoang FT Selection")
    
    grad_w = (loss_plus - loss_minus) / (2)
    fkg_instance.loss = (loss_plus+loss_minus)/2
    fkg_instance.backward(grad_w=grad_w)
    
    # Update weights after backward pass
    print(f"Loss of epoch {i+1}: ", fkg_instance.loss)
    
print("Best loss: ", fkg_instance.loss)

endTime = time.time()
print("Time to find best weight: ", endTime - startTime)
traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang FT Selection/FRB/TrainDataRule.csv'))
testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang FT Selection/FRB/TestDataRule.csv'))
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Hoang FT Selection")
print("--------------------------------")

e = [0.2,0.3]
r = [15,20]
for i in r:
    for j in e:
        print("__________Running FKG-S___________")
        traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang FT Selection/FRB/TrainDataRule.csv'))
        testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Hoang FT Selection/FRB/TestDataRule.csv'))
        base = [[int(float(x)) for x in row] for row in traindf.values]
        base = pd.DataFrame(base)
        test = [[int(float(x)) for x in row] for row in testdf.values]
        fkg_instance = FKGS()
        fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Hoang FT Selection",ran=i,e=j,folderPath=project_root)
        print("-"*100)

