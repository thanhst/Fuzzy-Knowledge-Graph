from module.FIS.FIS import FIS
from module.FKG.FKG_model import FKG
from module.FKG.FKG_S import FKGS
import pandas as pd

print("Only Image Feature")
print("__________Running Processing___________")
from module.Processing_Data import Image_Processing


print("__________Running FIS___________")
FIS(fileName="Only Image Feature",
    filePath=".\data\Dataset\OnlyImageFeature.csv",
    cluster=[5,5,5,5,5,5,5,5,5,5,6])
print("--------------------------------")

print("__________Running FKG___________")
traindf = pd.read_csv('./data/FIS/output/Only Image Feature/Rule_List.csv')
testdf = pd.read_csv('./data/FIS/output/Only Image Feature/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Only Image Feature")
print("--------------------------------")

print("__________Running FKG-S___________")
traindf = pd.read_csv('./data/FIS/output/Only Image Feature/Rule_List.csv')
testdf = pd.read_csv('./data/FIS/output/Only Image Feature/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKGS()
fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Only Image Feature",ran=20,e=0.2)
print("-"*100)
