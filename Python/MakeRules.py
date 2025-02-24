import numpy as np
import scipy.io as sio
from Python.module.FCM_Function import fcm_function as FCM_Function
from RuleWeight import RuleWeight

def MakeRules(train_dataset, var_data):
    train_data = train_dataset
    var_data = np.column_stack([train_data[:, 0], var_data])
    data_num, attribute_num = train_data.shape
    
    cluster = np.zeros(attribute_num)
    cluster[0] = 2
    
    # Determine the type of clustering for each feature
    for i in range(1, attribute_num):
        unique_vals = len(np.unique(train_data[:, i]))
        if unique_vals == 2:
            cluster[i] = 2
        elif unique_vals == 3:
            cluster[i] = 3
        else:
            cluster[i] = 5

    m = 2
    esp = 0.01
    maxTest = 200
    center_vector = [None] * attribute_num
    centers = [None] * (attribute_num - 1)
    centers_var = [None] * (attribute_num - 1)

    for feature_index in range(attribute_num-1):
        feature_data = train_data[:, feature_index]
        V = np.zeros(5)
        V_var = np.zeros(5)
        
        min_value = np.min(train_data[:, feature_index])
        max_value = np.max(train_data[:, feature_index])
        min_value_var = np.min(var_data[:, feature_index])
        max_value_var = np.max(var_data[:, feature_index])
        
        delta = max_value - min_value
        delta_var = max_value_var - min_value_var

        # Define fuzzy clustering ranges based on number of clusters
        if cluster[feature_index] == 2:
            V[0] = min_value - 0.5
            V[1] = max_value
            V_var[0] = min_value_var - 0.5
            V_var[1] = max_value_var
        elif cluster[feature_index] == 3:
            V[0] = min_value
            V[1] = min_value + delta / 2
            V[2] = max_value
            V_var[0] = min_value_var
            V_var[1] = min_value_var + delta_var / 2
            V_var[2] = max_value_var
        else:
            V[0] = min_value
            V[1] = min_value + delta / 4
            V[2] = min_value + 2 * delta / 4
            V[3] = min_value + 3 * delta / 4
            V[4] = max_value
            V_var[0] = min_value_var
            V_var[1] = min_value_var + delta_var / 4
            V_var[2] = min_value_var + 2 * delta_var / 4
            V_var[3] = min_value_var + 3 * delta_var / 4
            V_var[4] = max_value_var

        # Call the FCM_FunctTion for clustering
        centers[feature_index-1], U,_,__ = FCM_Function(feature_data, cluster[feature_index], V, m, esp, maxTest)
        centers_var[feature_index-1], U_var,_,__ = FCM_Function(feature_data, cluster[feature_index], V, m, esp, maxTest)
        
        center_vector[feature_index] = centers[feature_index-1]
        
        U=U.T
        U_var = U_var.T
        rules = np.zeros((data_num, attribute_num))
        rules_var = np.zeros((data_num, attribute_num))
        for i in range(data_num):
            maximum = np.max(U[i, :])
            maximum_var = np.max(U_var[i, :])
            for j in range(int(cluster[feature_index])):
                if maximum == U[i, j]:
                    rules[i, feature_index] = j
                if maximum_var == U_var[i, j]:
                    rules_var[i, feature_index] = j

    # Compute rule weights and filter
    t, sigma_M = RuleWeight(rules, train_data, cluster, center_vector)
    t_var, sigma_M_var = RuleWeight(rules_var, var_data, cluster, center_vector)
    
    # Adjusting sigma_M
    sigma_M[0, :] = np.zeros(sigma_M.shape[1])
    for i in range(attribute_num - 1):
        sigma_M[i, 1:5] = sigma_M[i, 0]

    sigma_M_var[0, :] = np.zeros(sigma_M_var.shape[1])
    for i in range(attribute_num - 1):
        sigma_M_var[i, 1:5] = sigma_M_var[i, 0]

    beta = np.zeros((data_num, attribute_num))
    for i in range(data_num):
        beta[i, :] = np.linalg.lstsq(train_data[i, 1:attribute_num].reshape(-1, 1), train_data[i, 0], rcond=None)[0]

    label = train_data[:, 0]
    
    # Filtering rules based on weight and similarity
    rules = np.delete(rules, 0, axis=1)
    rules_var = np.delete(rules_var, 0, axis=1)

    FilteredRules = []
    FilteredRules_var = []
    FilterT = []
    FilterT_var = []

    RuleCheck = np.zeros(attribute_num)
    j = 0

    for i in range(data_num):
        if not np.array_equal(rules[i, :], RuleCheck[0, :]):
            FilteredRules.append(np.concatenate([rules[i, :attribute_num-1], [label[i]]]))
            FilteredRules_var.append(rules_var[i, :])
            FilterT.append(t[i, :])
            FilterT_var.append(t_var[i, :])
            j += 1

    W_rule = np.min(FilterT, axis=1)
    W_rule_var = np.min(FilterT_var, axis=1)

    ruleList = np.array(FilteredRules)
    ruleList_var = np.array(FilteredRules_var)
    
    return sigma_M,sigma_M_var,ruleList,ruleList_var,centers,centers_var

    # Save results
    filename = train_dataset.replace('.txt', '.mat')
    filename = filename.replace('Database', 'FIS')
    
    sio.savemat(f'../output/{filename}', {'sigma_M': sigma_M, 'sigma_M_var': sigma_M_var})
    sio.savemat('../output/RuleList.mat', {'ruleList': ruleList, 'ruleList_var': ruleList_var})
    
    print('==============================================================================')
    print('Rule Generation process is done. RuleList.mat created.')
    print('==============================================================================')


# Example usage:
# MakeRules('your_train_dataset.txt', your_var_data)
