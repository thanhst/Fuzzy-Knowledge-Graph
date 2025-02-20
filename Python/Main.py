import numpy as np
from Rule2 import Rule2
from TestGraph2 import TestGraph2
import pandas as pd
from Run_Train_FIS import run_train_fis as Run_Train_FIS
from MakeRules import MakeRules
def main(database):
    a, b = database.shape
    
    label2 = database[database.iloc[:, 0] == 2].values
    label4 = database[database.iloc[:, 0] != 2].values
    
    folds_label2 = [label2[int(i*(len(label2)/10)):int((i+1)*(len(label2)/10))] for i in range(10)]
    
    folds_label4 = [label4[int(i*(len(label4)/10)):int((i+1)*(len(label4)/10))] for i in range(10)]
    
    RTrain2 = np.concatenate(folds_label2[:3])
    RVal2 = np.concatenate(folds_label2[3:5])
    
    
    RTrain4 = np.concatenate(folds_label4[:2])
    RVal4 = folds_label4[2]  
    
    RTrain = np.vstack((RTrain2, RTrain4))
    RVal = np.vstack((RVal2, RVal4))
    
    RTest2 = folds_label2[5]
    RTest4 = folds_label4[3]
    RTest = np.vstack((RTest2, RTest4))
    
    
    # centers = [np.mean(RTrain, axis=0)]
    # centers_var = [np.mean(RVal, axis=0)]
    
    num_attributes = RTrain.shape[1] - 1  # trừ cột target
    num_mfs = 3
    
    centers = []
    sigma_M = []
    for j in range(num_attributes):
        attr_data = RTrain[:, j+1]
        min_val = np.min(attr_data)
        max_val = np.max(attr_data)
        centers_j = np.linspace(min_val, max_val, num_mfs)  # hoặc dùng random
        centers.append(centers_j)
        if num_mfs > 1:
            distances = np.diff(centers_j)
            sigma_j = np.mean(distances) / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma_j = np.std(attr_data)
        sigma_M.append(np.full(num_mfs, sigma_j))
    centers = np.array(centers)
    sigma_M = np.array(sigma_M)
    
    # Tương tự, tính centers_var và sigma_M_var dựa trên RVal (hoặc dữ liệu var_data nếu có)
    centers_var = []
    sigma_M_var = []
    for j in range(num_attributes):
        attr_data_var = RVal[:, j+1]
        min_val_var = np.min(attr_data_var)
        max_val_var = np.max(attr_data_var)
        centers_var_j = np.linspace(min_val_var, max_val_var, num_mfs)
        centers_var.append(centers_var_j)
        if num_mfs > 1:
            distances_var = np.diff(centers_var_j)
            sigma_j_var = np.mean(distances_var) / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma_j_var = np.std(attr_data_var)
        sigma_M_var.append(np.full(num_mfs, sigma_j_var))
    centers_var = np.array(centers_var)
    sigma_M_var = np.array(sigma_M_var)
    
    ruleFuzzy = Rule2(RTrain)
    # sigma_M,sigma_M_var,rule_list,rule_list_var,centers,centers_var = MakeRules(RTrain,centers_var)
    df = pd.DataFrame(ruleFuzzy)
    df.to_csv("ruleFuzzy.csv", index=False, encoding="utf-8")
    
    # X1 = Run_Train_FIS(RTrain, RVal, centers, centers_var, ruleFuzzy,sigma_M,sigma_M_var)
    # print(X1)
    # AccuracyRVal =TestGraph2(RVal,A,B,X1,cluster,center_vector)
    # AccuracyRTest = TestGraph2(RTest, A, B, X1, cluster, center_vector)
database= pd.read_csv("./Tan1.txt")
main(database=database)
