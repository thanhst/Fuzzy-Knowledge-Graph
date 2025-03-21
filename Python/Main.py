from FIS import FIS
from FIS_raw_data import FIS_raw_data
from FIS_30  import FIS_30
# from module.FKG.FKG_model import FKG
# from module.FKG.FKG import FKG

import pandas as pd
# for i in range(10):
#     FIS_raw_data(Turn=i)
#     FIS(Turn = i)
#     FIS_30(Turn=i)
# FIS_raw_data()
    # cluster = [2,2,3,3,3,2,2,2,2,2,2,3,3,3,3,2,3,2,3,2,2,2,5,5,5,5,5,5,5,5,5,5,6]
print("Only Image Feature")
FIS(fileName="Only Image Feature",
    filePath="D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\Dataset\OnlyImageFeture.csv",
    cluster=[5,5,5,5,5,5,5,5,5,5,6])

# print("Only Table Feature Remove Missing")
# FIS(fileName="Only Table Feature Remove Missing",
#     filePath="D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\Dataset\OnlyTableFeatureRemoveMissing.csv",
#     cluster=[3,5,2,3,2,3,2,2,2,6])

# print("Only Table Feature Remove Missing")

# FIS(fileName="Only Table Feature Replace Missing",
#     filePath="D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\Dataset\OnlyTableFeatureReplaceMissing.csv",
#     cluster=[2,2,5,5,3,2,2,2,2,2,2,3,5,3,3,2,3,2,3,2,2,2,6])

# print("Fusion Feature Remove Missing")


# FIS(fileName="Fusion Feature Remove Missing",
#     filePath="D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\Dataset\FusionFeatureRemoveMissing.csv",
#     cluster=[3,5,2,3,2,3,2,2,2,5,5,5,5,5,5,5,5,5,5,6])

# print("Fusion Feature Replace Missing")
# FIS(fileName="Fusion Feature Replace Missing",
#     filePath="D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\Dataset\FusionFeatureReplaceMissing.csv",
#     cluster = [2,2,3,3,3,2,2,2,2,2,2,3,3,3,3,2,3,2,3,2,2,2,5,5,5,5,5,5,5,5,5,5,6])


from module.FKG.FKG import FKG
# from module.FKG.FKG_model import FKG
traindf = pd.read_csv('./data/Dataset/OnlyImageFeture.csv')
# numerical_cols = traindf.select_dtypes(include=['number']).columns
# for col in numerical_cols:
#     traindf[col].fillna(traindf[col].median(), inplace=True)
traindf = traindf.values

base = [[int(float(x)) for x in row] for row in traindf]
base = pd.DataFrame(base)
fkg_instance = FKG()
traindata = pd.read_csv("./data/FIS/output/Test/train.csv")
testdata = pd.read_csv("./data/FIS/output/Test/test.csv")
fkg_instance.FKG_test(train=traindata,test=testdata,Turn=None,Modality="Only Image Feature")
# train=pd.read_csv
# fkg_instance.FKG_test(train=,test=,Turn=None,Modality="Only Image Feture")

# traindf = pd.read_csv('./data/FIS/output/Only Table Feature Remove Missing/Rule_List.csv')
# traindf = traindf.values
# base = [[int(float(x)) for x in row] for row in traindf]
# base = pd.DataFrame(base)
# fkg_instance = FKG()
# fkg_instance.FKG(df = base,Turn=None,Modality="Only Table Feature Remove Missing")

# traindf = pd.read_csv('./data/FIS/output/Only Table Feature Replace Missing/Rule_List.csv')
# traindf = traindf.values
# base = [[int(float(x)) for x in row] for row in traindf]
# base = pd.DataFrame(base)
# fkg_instance = FKG()
# fkg_instance.FKG(df = base,Turn=None,Modality="Only Table Feature Replace Missing")

# traindf = pd.read_csv('./data/FIS/output/Fusion Feature Remove Missing/Rule_List.csv')
# traindf = traindf.values
# base = [[int(float(x)) for x in row] for row in traindf]
# base = pd.DataFrame(base)
# fkg_instance = FKG()
# fkg_instance.FKG(df = base,Turn=None,Modality="Fusion Feature Remove Missing")

# traindf = pd.read_csv('./data/FIS/output/Fusion Feature Replace Missing/Rule_List.csv')
# traindf = traindf.values
# base = [[int(float(x)) for x in row] for row in traindf]
# base = pd.DataFrame(base)
# fkg_instance = FKG()
# fkg_instance.FKG(df = base,Turn=None,Modality="Fusion Feature Replace Missing")