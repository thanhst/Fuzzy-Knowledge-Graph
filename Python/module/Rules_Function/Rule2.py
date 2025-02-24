
import numpy as np
from Python.module.FCM_Function import fcm_function as FCM_Function
from Python.module.RulesFunction.RuleWeight import RuleWeight

def Rule2(train_data):
    data_num, attribute_num = train_data.shape
    data = train_data[:, 1:]
    nhan1 = train_data[:, 0]
    
    cluster = np.zeros(attribute_num, dtype=int)
    for i in range(attribute_num):
        temp = len(np.unique(train_data[:, i]))
        cluster[i] = min(temp, 3)
    
    m = 2
    esp = 0.01
    maxTest = 200
    center_vector = [None] * attribute_num
    centers = [None] * (attribute_num - 1)
    
    rules = np.zeros((data_num, attribute_num), dtype=int)
    for feature_index in range(attribute_num):
        feature_data = train_data[:, feature_index]
        min_value = np.min(feature_data)
        max_value = np.max(feature_data)
        delta = max_value - min_value
        
        if cluster[feature_index] == 2:
            V = np.array([[min_value - 0.5], [max_value]])
        elif cluster[feature_index] == 3:
            V = np.array([[min_value], [min_value + delta / 2], [max_value]])
        else:
            V = np.array([
                [min_value], [min_value + delta / 4], [min_value + 2 * delta / 4], 
                [min_value + 3 * delta / 4], [max_value]
            ])
        
        center, U,_,__ = FCM_Function(feature_data, cluster[feature_index], V, m, esp, maxTest)
        U = U.T
        center_vector[feature_index] = center[:, 0]
        
        for i in range(data_num):
            max_val = np.max(U[i, :])
            for j in range(cluster[feature_index]):
                if max_val == U[i, j]:
                    rules[i, feature_index] = j + 1
        
        if feature_index != 0:
            centers[feature_index - 1] = center[:, 0]
    
    t, sigma_M = RuleWeight(rules, train_data, cluster, center_vector)
    sigma_M = sigma_M.reshape(-1,1)
    sigma_M = sigma_M[1:, :]
    for i in range(attribute_num - 1):
        sigma_M[i, 1:5] = sigma_M[i, 0]
    
    beta = np.zeros((data_num, attribute_num))
    for i in range(data_num):
        X = np.hstack(([1], train_data[i, 1:])).reshape(1, -1) 
        beta[i, :] = np.linalg.lstsq(X, train_data[i, 0].reshape(-1, 1), rcond=None)[0].flatten()
    for i in range(data_num):
        rules[i, attribute_num-1] = np.min(t[i, 1:])
    valid_rule = np.ones(data_num, dtype=bool)
    for i in range(data_num - 1):
        for j in range(i + 1, data_num):
            if np.array_equal(rules[i, 1:attribute_num-1], rules[j, 1:attribute_num-1]):
                if rules[i, attribute_num-1] > rules[j, attribute_num-1]:
                    valid_rule[j] = False
                else:
                    valid_rule[i] = True
                    
    rules = rules[valid_rule, :]
    rules = np.delete(rules, 0, axis=1)
    rule_check = np.zeros((1, attribute_num))
    filtered_rules = []
    filter_t = []
    
    for i in range(data_num):
        if not np.array_equal(rules[i, :], rule_check[0, :]):
            filtered_rules.append(np.hstack((nhan1[i], rules[i, :attribute_num - 1])))
            filter_t.append(t[i, :])
    filtered_rules = np.array(filtered_rules)
    filter_t = np.array(filter_t)
    
    W_rule = np.min(filter_t, axis=1)
    rule_list = filtered_rules
    r1, r2 = rule_list.shape
    w = np.zeros((r2, 3))
    
    for i in range(1, r2):
        dem = np.zeros(3)
        for j in range(r1):
            for l in range(cluster[i]):
                if rule_list[j, i] == l + 1:
                    dem[l] += 1
        w[i, :] = dem / r1
    
    dem2 = np.sum(rule_list[:, 0] == 2)
    dem4 = np.sum(rule_list[:, 0] != 2)
    wl = np.array([dem2 / r1, dem4 / r1])
    
    m = np.zeros((r1, r2 - 1))
    for i in range(1, r2 - 1):
        for j in range(r1):
            m[j, i] = w[i, int(rule_list[j, i]) - 1] * w[i + 1, int(rule_list[j, i + 1]) - 1]
    
    for j in range(r1):
        if rule_list[j, 0] == 2:
            m[j, 0] = wl[0] * w[r2 - 1, int(rule_list[j, r2 - 1]) - 1]
        else:
            m[j, 0] = wl[1] * w[r2 - 1, int(rule_list[j, r2 - 1]) - 1]
    
    m1 = np.sum(m, axis=1)
    tbc = np.mean(m1)
    Rcur = np.zeros((r1, rule_list.shape[1]))
    dem = 0
    
    for i in range(r1):
        if m1[i] > tbc:
            Rcur[dem] = (rule_list[i, :])
            dem += 1

    Rcur = np.array(Rcur)
    return Rcur
