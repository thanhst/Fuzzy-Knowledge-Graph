import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'Source_code')
import pandas as pd
from module.FKG.FKG_general import FKG

script_dir = os.path.dirname(os.path.abspath('test_fkg_full.py'))
project_root = os.path.join(script_dir, 'Source_code')
print('Project root:', project_root)

train_path = os.path.join(project_root, 'data', 'FIS', 'output', 'Diabetic Retinopathy Feature', 'FRB', 'TrainDataRule.csv')
test_path = os.path.join(project_root, 'data', 'FIS', 'output', 'Diabetic Retinopathy Feature', 'FRB', 'TestDataRule.csv')

print('Loading training data from:', train_path)
traindf = pd.read_csv(train_path)
print('Training data shape:', traindf.shape)

print('Loading test data from:', test_path)
testdf = pd.read_csv(test_path)
print('Test data shape:', testdf.shape)

print('Converting data...')
base = [[int(float(x)) for x in row] for row in traindf.values]
base = pd.DataFrame(base)
test = [[int(float(x)) for x in row] for row in testdf.values]

print('Training FKG with full dataset...')
fkg_instance = FKG()
fkg_instance.FKG(df=base, testdf=test, Turn=None, Modality='Diabetic Retinopathy Feature')
print('FKG completed successfully!')
