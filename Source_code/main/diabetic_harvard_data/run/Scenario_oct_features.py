import sys
import os

# Lấy đường dẫn tuyệt đối tới thư mục gốc của project (ở đây là Source_code)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..",".."))  # lên 2 cấp

if project_root not in sys.path:
    sys.path.append(project_root)

from module.FIS.FIS import FIS
from module.FKG.FKG_general import FKG
from module.FKG.FKG_S import FKGS
import pandas as pd

print("Dataset Multimodality oct image")
print("__________Running Processing___________")
# from module.Processing_Data import Image_Processing


print("__________Running FIS___________")
FIS(fileName="Dataset Multimodality oct image",
    filePath=".\main\diabetic_harvard_data\data\\OCT_features\\images_ft_data.csv",
    cluster=[5,5,5,5,5,5,5,5,5,5,2])
print("--------------------------------")

print("__________Running FKG___________")
traindf = pd.read_csv('./data/FIS/output/Dataset Multimodality oct image/FRB/TrainDataRule.csv')
testdf = pd.read_csv('./data/FIS/output/Dataset Multimodality oct image/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Dataset Multimodality oct image")
print("--------------------------------")
e = [0.2,0.3]
r = [15,20]
for i in r:
    for j in e:
        print("__________Running FKG-S___________")
        traindf = pd.read_csv('./data/FIS/output/Dataset Multimodality oct image/FRB/TrainDataRule.csv')
        testdf = pd.read_csv('./data/FIS/output/Dataset Multimodality oct image/FRB/TestDataRule.csv')
        base = [[int(float(x)) for x in row] for row in traindf.values]
        base = pd.DataFrame(base)
        test = [[int(float(x)) for x in row] for row in testdf.values]
        fkg_instance = FKGS()
        fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Dataset Multimodality oct image",ran=i,e=j)
        print("-"*100)
