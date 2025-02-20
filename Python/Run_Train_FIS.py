import numpy as np
import time

def gaussmf(x, params):
    sigma, center = params
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def GaussMF(x1, label, MFnumber, sigma, centers):
    for i in range(int(MFnumber)):
        if label == i + 1:
            result = gaussmf(x1, [sigma, centers[0]])
            return result
    return 0


def run_train_fis(temp_data, var_data, centers, centers_var, ruleList,sigma_M,sigma_M_var):
    eta = 0.000001
    term_num = 5
    train_output = temp_data[:, 0]
    temp_data = np.delete(temp_data, 0, axis=1)
    train_input = temp_data
    
    input_num = train_input.shape[0]
    attri_num = train_input.shape[1]
    rule_num = ruleList.shape[0]
    rule_length = ruleList.shape[1]
    
    re_degree_M = np.ones((input_num, attri_num, term_num))
    im_degree_M = np.ones((input_num, attri_num, term_num))
    wtsum_re = np.zeros(input_num)
    wtsum_im = np.zeros(input_num)
    M_DataPerRule_re = np.zeros((input_num, rule_num))
    M_DataPerRule_im = np.zeros((input_num, rule_num))

    start_time_train = time.time()
    
    for i in range(input_num):
        for j in range(attri_num):
            for k in range(len(centers[j])):
                re_degree_M[i, j, k] = gaussmf(train_input[i, j], [sigma_M[j, k], centers[j][k]])
                im_degree_M[i, j, k] = abs(-(gaussmf(var_data[i, j], [sigma_M_var[j, k], centers_var[j][k]])) * (var_data[i, j] - centers_var[j][k]) / (sigma_M_var[j, k] ** 2))

    result = np.zeros(input_num)
    result_re = np.zeros(input_num)
    result_im = np.zeros(input_num)

    for i in range(input_num):
        result_re[i] = 0
        result_im[i] = 0
        wtsum_re[i] = 0
        wtsum_im[i] = 0
        for j in range(rule_num):
            wtsum_re[i] += M_DataPerRule_re[i, j] * np.cos(M_DataPerRule_im[i, j])
            wtsum_im[i] += M_DataPerRule_re[i, j] * np.sin(M_DataPerRule_im[i, j])
            result_re[i] += M_DataPerRule_re[i, j] * np.cos(M_DataPerRule_im[i, j]) * ruleList[j, rule_length - 1]
            result_im[i] += M_DataPerRule_re[i, j] * np.sin(M_DataPerRule_im[i, j]) * ruleList[j, rule_length - 1]

        result[i] = np.sqrt((result_re[i] ** 2 + result_im[i] ** 2) / (wtsum_re[i] ** 2 + wtsum_im[i] ** 2))

    # Determine if the result is close to max or min train output
    temp = np.copy(result)
    for i in range(len(temp)):
        if temp[i] >= 0.41 * (np.max(train_output) + np.min(train_output)):
            temp[i] = np.max(train_output)
        else:
            temp[i] = np.min(train_output)

    end_time_train = time.time()
    time_elapsed = end_time_train - start_time_train

    # Compute confusion matrix
    TP = np.sum((temp == train_output) & (temp == 0))  # True Positives for class 0
    TN = np.sum((temp == train_output) & (temp == 1))  # True Negatives for class 1
    FN = np.sum((train_output == 0) & (temp == 1))    # False Negatives
    FP = np.sum((train_output == 1) & (temp == 0))    # False Positives

    accuracy = (TN + TP) / (TN + TP + FN + FP)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    return time_elapsed, accuracy, recall, precision

# Example usage:
# Define your input variables, temp_data, var_data, centers, centers_var, ruleList, and ruleList_var
# Then call the function:
# time, accuracy, recall, precision = run_train_fis(temp_data, var_data, centers, centers_var, ruleList, ruleList_var)
