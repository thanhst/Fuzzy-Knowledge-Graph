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

print("Diabetic Retinopathy Image Feature")

print("__________Running Processing___________")
# from module.Processing_Data import Diabetic_image_processing

print("__________Running FIS___________")
FIS(fileName="Diabetic Retinopathy Image Feature",
    filePath=os.path.join(project_root,"data\Dataset_diabetic\Image_feature\data_process.csv"),
    cluster=[5,5,5,5,5,5,5,5,5,5,2])
print("--------------------------------")

print("__________Running FKG___________")
traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Image Feature/Rule_List.csv'))
testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Image Feature/FRB/TestDataRule.csv'))
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Image Feature")
print("--------------------------------")

print("__________Running FKG-S___________")
traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Image Feature/Rule_List.csv'))
testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Diabetic Retinopathy Image Feature/FRB/TestDataRule.csv'))
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKGS()
fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Image Feature",ran=20,e=0.2)
print("-"*100)

