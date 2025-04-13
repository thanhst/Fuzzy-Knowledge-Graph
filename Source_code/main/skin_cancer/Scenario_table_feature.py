import sys
import os

from module.FIS.FIS import FIS
from module.FKG.FKG_general import FKG
import pandas as pd
from module.FKG.FKG_S import FKGS

print("Only Table Feature Remove Missing")
print("__________Running Processing___________")
from module.Processing_Data import Metadata_processing

print("__________Running FIS___________")
FIS(fileName="Only Table Feature Remove Missing",
    filePath=".\data\Dataset\OnlyTableFeatureRemoveMissing.csv",
    cluster=[3,5,2,3,2,3,2,2,2,6])
print("--------------------------------")

print("__________Running FKG___________")
traindf = pd.read_csv('./data/FIS/output/Only Table Feature Remove Missing/Rule_List.csv')
testdf = pd.read_csv('./data/FIS/output/Only Table Feature Remove Missing/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Only Table Feature Remove Missing")
print("--------------------------------")

print("__________Running FKG-S___________")
traindf = pd.read_csv('./data/FIS/output/Only Table Feature Remove Missing/Rule_List.csv')
testdf = pd.read_csv('./data/FIS/output/Only Table Feature Remove Missing/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKGS()
fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Only Table Feature Remove Missing",ran=10,e=0.2)
print("-"*100)
