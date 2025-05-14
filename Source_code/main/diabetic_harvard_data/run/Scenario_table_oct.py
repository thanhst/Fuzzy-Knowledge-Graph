import sys
import os

# Lấy đường dẫn tuyệt đối tới thư mục gốc của project (ở đây là Source_code)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # lên 2 cấp

if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
# from module.FIS.FIS import FIS
# from module.FKG.FKG_general import FKG
# from module.FKG.FKG_S import FKGS
from sklearn.model_selection import KFold
import numpy as np

print("Diabetic Retinopathy Fusion Feature Filter")

print("__________Running Processing___________")
from process.process_image import preprocessing_image,preprocessing_image_oct
from process.process_data import preprocessing_data,process_corr_advanced
# preprocessing_image_oct(folder_save='OCT_features',file_path='D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Source_code\data\Dataset3modal\Test')
# preprocessing_image(folder_save='SOL_features',file_path='D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Source_code\data\Dataset3modal\Test')
# preprocessing_data(folder_save='table_features',file_path='D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Source_code\data\Dataset3modal\Test')
process_corr_advanced(folder_save='table_features',file_path='D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Source_code\main\diabetic_harvard_data\data\\table_features\\table_ft_data.csv')

