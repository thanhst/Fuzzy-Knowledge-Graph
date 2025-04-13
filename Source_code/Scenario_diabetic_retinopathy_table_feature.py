from module.FIS.FIS import FIS
from module.FKG.FKG_general import FKG
from module.FKG.FKG_S import FKGS

import pandas as pd

print("Diabetic Retinopathy Feature")

print("__________Running Processing___________")
from module.Processing_Data import Diabetic_metadata_processing

print("__________Running FIS___________")
FIS(fileName="Diabetic Retinopathy Metadata Feature",
    filePath=".\data\Dataset_diabetic\Metadata_feature\data_process.csv",
    cluster=[3,2,4,2,2,2,2,2,2,2,2,2,2,2])
print("--------------------------------")

print("__________Running FKG___________")
traindf = pd.read_csv('./data/FIS/output/Diabetic Retinopathy Metadata Feature/Rule_List.csv')
testdf = pd.read_csv('./data/FIS/output/Diabetic Retinopathy Metadata Feature/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Metadata Feature")
print("--------------------------------")

print("__________Running FKG-S___________")
traindf = pd.read_csv('./data/FIS/output/Diabetic Retinopathy Metadata Feature/Rule_List.csv')
testdf = pd.read_csv('./data/FIS/output/Diabetic Retinopathy Metadata Feature/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKGS()
fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Diabetic Retinopathy Metadata Feature",ran=20,e=0.2)
print("-"*100)

