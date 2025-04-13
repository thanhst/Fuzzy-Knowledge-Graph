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

print("Fusion Feature Remove Missing")

print("__________Running Processing___________")
from module.Processing_Data import Fusion_processing

print("__________Running FIS___________")
FIS(fileName="Fusion Feature Remove Missing",
    filePath=os.path.join(project_root,"data\Dataset\FusionFeatureRemoveMissing.csv"),
    cluster=[3,5,2,3,2,3,2,2,2,5,5,5,5,5,5,5,5,5,5,6])
print("--------------------------------")

print("__________Running FKG___________")
traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Fusion Feature Remove Missing/Rule_List.csv'))
testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Fusion Feature Remove Missing/FRB/TestDataRule.csv'))
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Fusion Feature Remove Missing")
print("--------------------------------")

print("__________Running FKG-S___________")
traindf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Fusion Feature Remove Missing/Rule_List.csv'))
testdf = pd.read_csv(os.path.join(project_root,'data/FIS/output/Fusion Feature Remove Missing/FRB/TestDataRule.csv'))
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKGS()
fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Fusion Feature Remove Missing",ran=20,e=0.2)
print("-"*100)
