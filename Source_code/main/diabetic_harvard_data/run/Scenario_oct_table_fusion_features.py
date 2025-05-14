import pandas as pd
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..",".."))  # lên 2 cấp

if project_root not in sys.path:
    sys.path.append(project_root)
    
from sklearn.preprocessing import MinMaxScaler
from module.Processing_Data.Fusion_function import feature_selection
from module.Processing_Data.Fusion_function import filter_multimodal_selection,hadamard_selection,tensor_selection,wrapper_multimodal_feature_selection

df_oct= pd.read_csv(os.path.join(project_root,'main/diabetic_harvard_data/data/OCT_features/images_ft.csv'))
df_table = pd.read_csv(os.path.join(project_root,'main/diabetic_harvard_data/data/table_features/table_ft.csv'))
Label = df_table.iloc[:, -1]
Fimg = df_oct.iloc[:,1:-1]
Ftab = df_table.iloc[:,1:-1]

# Fimg_select = pd.DataFrame(feature_selection.select_features(X=Fimg,y = Label, k = 6))
# Ftab_select = pd.DataFrame(feature_selection.select_features(X=Ftab,y = Label, k = 4))
# merged_features = pd.concat([Fimg_select, Ftab_select], axis=1)

#hadamard
# merged_features = pd.DataFrame(hadamard_selection.hadamard_fusion(Fimg=Fimg,Ftab=Ftab,common_dim=4))


#filter_multimodal_selection
Fimg = Fimg.to_numpy()
Ftab = Ftab.to_numpy()
merged_features = pd.DataFrame(filter_multimodal_selection.filter_multimodal_selection(Fimg=Fimg,Ftab=Ftab,target=Label,k_img=6,k_tab=4))


scaler = MinMaxScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(merged_features), columns=merged_features.columns)
final_data = pd.concat([features_scaled, Label], axis=1)
final_data.to_csv(os.path.join(project_root,'main/diabetic_harvard_data/data/oct_table_fusion_ft.csv'),index=False)

from module.FIS.FIS import FIS
from module.FKG.FKG_general import FKG
from module.FKG.FKG_S import FKGS

print('Dataset Multimodality oct fusion table')

print("__________Running FIS___________")
FIS(fileName="Dataset Multimodality oct fusion table",
    filePath=".\main\diabetic_harvard_data\data\\oct_table_fusion_ft.csv",
    # cluster=[5,5,5,5,5,5,3,2,2,3,2])
    cluster=[5,5,5,5,5,5,3,2,2,22,2])
print("--------------------------------")

print("__________Running FKG___________")
traindf = pd.read_csv('./data/FIS/output/Dataset Multimodality oct fusion table/Rule_List.csv')
testdf = pd.read_csv('./data/FIS/output/Dataset Multimodality oct fusion table/FRB/TestDataRule.csv')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]
fkg_instance = FKG()
fkg_instance.FKG(df = base,testdf=test,Turn=None,Modality="Dataset Multimodality oct fusion table")
print("--------------------------------")
e = [0.2,0.3]
r = [15,20]
for i in r:
    for j in e:
        print("__________Running FKG-S___________")
        traindf = pd.read_csv('./data/FIS/output/Dataset Multimodality oct fusion table/Rule_List.csv')
        testdf = pd.read_csv('./data/FIS/output/Dataset Multimodality oct fusion table/FRB/TestDataRule.csv')
        base = [[int(float(x)) for x in row] for row in traindf.values]
        base = pd.DataFrame(base)
        test = [[int(float(x)) for x in row] for row in testdf.values]
        fkg_instance = FKGS()
        fkg_instance.FKGS(df = base,testdf=test,Turn=None,Modality="Dataset Multimodality oct fusion table",ran=i,e=j)
        print("-"*100)