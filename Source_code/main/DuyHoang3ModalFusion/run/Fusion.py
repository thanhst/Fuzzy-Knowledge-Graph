import pandas as pd
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..",".."))  # lên 2 cấp

if project_root not in sys.path:
    sys.path.append(project_root)
    
from sklearn.preprocessing import MinMaxScaler

# Load từ processed features (CSV)
# Load từ processed features (CSV)
# Có thể chọn split: 'train', 'val', 'test', 'val_retrieval'
split = 'train'
# Đường dẫn: từ Source_code/main/DuyHoang3ModalFusion/run lên 4 cấp đến workspace root, rồi vào Combination/csv_features/
csv_path = os.path.join(current_dir, '..', '..', '..', '..', 'Combination', 'csv_features', f'processed_features_{split}.csv')

# Normalize path
csv_path = os.path.abspath(csv_path)

print(f"\n{'='*60}")
print(f"Loading fused features from: {csv_path}")
print(f"{'='*60}")

# Check if file exists
if not os.path.exists(csv_path):
    print(f"\n❌ ERROR: File not found!")
    print(f"  Expected path: {csv_path}")
    print(f"\n  Current directory: {current_dir}")
    print(f"\n  Available files in Combination folder:")
    combination_dir = os.path.join(current_dir, '..', '..', 'Combination')
    if os.path.exists(combination_dir):
        for f in os.listdir(combination_dir):
            print(f"    - {f}")
    sys.exit(1)

df_features = pd.read_csv(csv_path)
print(f"✓ File loaded successfully!")

# Tách features và label
Label = df_features.iloc[:, -1]  # Cột cuối là label
hadm_id = df_features.iloc[:, 0]  # hadm_id ở cột đầu
fused_features = df_features.iloc[:, 1:-1]  # Fused features (CXR + ECG + Labs)

num_features = fused_features.shape[1]

print(f"\n✓ Data loaded successfully!")
print(f"  - Total samples: {len(Label)}")
print(f"  - Fused features: {num_features}")
print(f"  - Label distribution: {Label.value_counts().to_dict()}")

# Calculate cluster size based on number of features
# Each cluster typically has 5 features, with remainder as last cluster
cluster_size = 5
num_clusters = num_features // cluster_size
remainder = num_features % cluster_size
cluster = [cluster_size] * num_clusters
if remainder > 0:
    cluster.append(remainder)

print(f"\n✓ Auto-calculated cluster: {cluster}")
print(f"  - Total clusters: {len(cluster)}")

# Use the CSV directly (already normalized)
output_csv = csv_path
print(f"\n✓ Using data from: {output_csv}\n")

from module.FIS.FIS import FIS
from module.FKG.FKG_general import FKG
from module.FKG.FKG_S import FKGS

dataset_name = f"Symile_MIMIC_Fused_{split.upper()}"

print(f"\n{'='*60}")
print(f"Dataset: {dataset_name}")
print(f"{'='*60}")

print("\n__________Running FIS___________")
FIS(fileName=dataset_name,
    filePath=output_csv,
    cluster=cluster)  # Auto-calculated from number of features
print("--------------------------------")

print("\n__________Running FKG___________")
fis_output_dir = os.path.join(current_dir, 'data', 'FIS', 'output', dataset_name)
traindf_path = os.path.join(fis_output_dir, 'Rule_List.csv')
testdf_path = os.path.join(fis_output_dir, 'FRB', 'TestDataRule.csv')

if os.path.exists(traindf_path) and os.path.exists(testdf_path):
    traindf = pd.read_csv(traindf_path)
    testdf = pd.read_csv(testdf_path)
    base = [[int(float(x)) for x in row] for row in traindf.values]
    base = pd.DataFrame(base)
    test = [[int(float(x)) for x in row] for row in testdf.values]
    fkg_instance = FKG()
    fkg_instance.FKG(df=base, testdf=test, Turn=None, Modality=dataset_name)
    print("--------------------------------")
    
    print("\n__________Running FKG-S___________")
    e = [0.2, 0.3]
    r = [15, 20]
    for i in r:
        for j in e:
            print(f"\n  [FKG-S] ran={i}, e={j}")
            traindf = pd.read_csv(traindf_path)
            testdf = pd.read_csv(testdf_path)
            base = [[int(float(x)) for x in row] for row in traindf.values]
            base = pd.DataFrame(base)
            test = [[int(float(x)) for x in row] for row in testdf.values]
            fkg_instance = FKGS()
            fkg_instance.FKGS(df=base, testdf=test, Turn=None, Modality=dataset_name, ran=i, e=j, folderPath=project_root)
    print("-"*100)
    print("\n\u2713 Pipeline completed!")
else:
    print(f"\n\u26a0 FIS output not found. Please ensure FIS ran successfully.")
    print(f"  - Expected: {traindf_path}")
    print(f"  - Expected: {testdf_path}")