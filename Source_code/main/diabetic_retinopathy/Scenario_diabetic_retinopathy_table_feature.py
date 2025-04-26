import sys
import os

# Lấy đường dẫn tuyệt đối tới thư mục gốc của project (ở đây là Source_code)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # lên 2 cấp

if project_root not in sys.path:
    sys.path.append(project_root)
    
from module.FIS.FIS import FIS
from module.FKG.FKG_general import FKG
from module.FKG.FKG_S import FKGS

import pandas as pd

print("Diabetic Retinopathy Table Feature")

print("__________Running Processing___________")
from module.Processing_Data import Diabetic_metadata_processing
# Diabetic_metadata_processing.processing_wrapper(file_path_table="data/Dataset_diabetic/data_process.csv",file_path_img="data/Image/fundus_photo",folder_save="Metadata_feature",max_img=5,max_tab=4)
Diabetic_metadata_processing.processing_ft_selection(file_path_table="data/Dataset_diabetic/data_process.csv",file_path_img="data/Image/fundus_photo",folder_save="Metadata_feature",k=13)
print("__________Running FIS___________")
FIS(fileName="Diabetic Retinopathy Metadata Feature",
    filePath=os.path.join(project_root,'data\Dataset_diabetic\Metadata_feature\data_process.csv'),
    cluster=[3,2,4,2,2,2,2,2,2,2,2,2,2,2])
    # cluster=[5,5,5,5,5,5,5,5,5,2])
print("--------------------------------")

print("__________Running FKG___________")
traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Metadata Feature/Rule_List_All.csv'))
testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Metadata Feature/FRB/TestDataRule.csv'))
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Metadata Feature")

print("--------------------------------")
e = [0.2,0.3]
r = [15,20]
for i in r:
    for j in e:
        print("__________Running FKG-S___________")
        traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Metadata Feature/Rule_List.csv'))
        testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Metadata Feature/FRB/TestDataRule.csv'))
        base = [[int(float(x)) for x in row] for row in traindf.values]
        base = pd.DataFrame(base)
        test = [[int(float(x)) for x in row] for row in testdf.values]
        fkg_instance = FKGS()
        fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Metadata Feature",ran=i,e=j)
        print("-"*100)

