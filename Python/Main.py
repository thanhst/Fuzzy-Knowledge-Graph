from FIS import FIS
from FIS_raw_data import FIS_raw_data
from FIS_30  import FIS_30
# from module.FKG.FKG_model import FKG
# from module.FKG.FKG import FKG
from module.FKG.FKG_model import FKG

import pandas as pd
# for i in range(10):
#     FIS_raw_data(Turn=i)
#     FIS(Turn = i)
#     FIS_30(Turn=i)
# FIS_raw_data()
    # cluster = [2,2,3,3,3,2,2,2,2,2,2,3,3,3,3,2,3,2,3,2,2,2,5,5,5,5,5,5,5,5,5,5,6]
print("Only Image Feature")
FIS(fileName="Only Image Feature",
    filePath="D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\Dataset\OnlyImageFeature.csv",
    cluster=[5,5,5,5,5,5,5,5,5,5,6])

traindf = pd.read_csv('./data/FIS/output/Only Image Feature/Rule_List.csv')
testdf = pd.read_csv('./data/FIS/output/Only Image Feature/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Only Image Feature")
# numerical_cols = traindf.select_dtypes(include=['number']).columns
# for col in numerical_cols:
#     traindf[col].fillna(traindf[col].median(), inplace=True)
# traindf = traindf.values



print("Only Table Feature Remove Missing")

FIS(fileName="Only Table Feature Remove Missing",
    filePath="D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\Dataset\OnlyTableFeatureRemoveMissing.csv",
    cluster=[3,5,2,3,2,3,2,2,2,6])
traindf = pd.read_csv('./data/FIS/output/Only Table Feature Remove Missing/Rule_List.csv')
testdf = pd.read_csv('./data/FIS/output/Only Table Feature Remove Missing/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Only Table Feature Remove Missing")

# print("Only Table Feature Replace Missing")

# # FIS(fileName="Only Table Feature Replace Missing",
# #     filePath="D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\Dataset\OnlyTableFeatureReplaceMissing.csv",
# #     cluster=[2,2,5,5,3,2,2,2,2,2,2,3,5,3,3,2,3,2,3,2,2,2,6])

# traindf = pd.read_csv('./data/FIS/output/Only Table Feature Replace Missing/Rule_List.csv')
# traindf = traindf.values
# base = [[int(float(x)) for x in row] for row in traindf]
# base = pd.DataFrame(base)
# fkg_instance = FKG()
# fkg_instance.FKG(df = base,Turn=None,Modality="Only Table Feature Replace Missing")

print("Fusion Feature Remove Missing")


FIS(fileName="Fusion Feature Remove Missing",
    filePath="D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\Dataset\FusionFeatureRemoveMissing.csv",
    cluster=[3,5,2,3,2,3,2,2,2,5,5,5,5,5,5,5,5,5,5,6])
traindf = pd.read_csv('./data/FIS/output/Fusion Feature Remove Missing/Rule_List.csv')
testdf = pd.read_csv('./data/FIS/output/Fusion Feature Remove Missing/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Fusion Feature Remove Missing")

# print("Fusion Feature Replace Missing")
# # FIS(fileName="Fusion Feature Replace Missing",
# #     filePath="D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\Dataset\FusionFeatureReplaceMissing.csv",
# #     cluster = [2,2,3,3,3,2,2,2,2,2,2,3,3,3,3,2,3,2,3,2,2,2,5,5,5,5,5,5,5,5,5,5,6])

# traindf = pd.read_csv('./data/FIS/output/Fusion Feature Replace Missing/Rule_List.csv')
# traindf = traindf.values
# base = [[int(float(x)) for x in row] for row in traindf]
# base = pd.DataFrame(base)
# fkg_instance = FKG()
# fkg_instance.FKG(df = base,Turn=None,Modality="Fusion Feature Replace Missing")

# from module.FKG.FKG import FKG

# base = [[int(float(x)) for x in row] for row in traindf]
# base = pd.DataFrame(base)
# fkg_instance = FKG()
# fkg_instance.FKG()

# traindata = pd.read_csv("./data/FIS/output/Test/train_int.csv")
# testdata = pd.read_csv("./data/FIS/output/Test/test_int.csv")
# traindata = [[int(float(x)) for x in row] for row in traindata.values]
# traindata = pd.DataFrame(traindata)
# testdata = [[int(float(x)) for x in row] for row in testdata.values]
# testdata = pd.DataFrame(testdata)
# fkg_instance = FKG()
# fkg_instance.FKG_test(train=traindata,test=testdata,Turn=None,Modality="Only Image Feature")






