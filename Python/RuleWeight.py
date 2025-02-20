import numpy as np
from scipy.stats import norm

def compute_sigma(center_vector):
    d = 0
    if len(center_vector) == 2:
        d = abs(center_vector[0] - center_vector[1])
    else:
        for i in range(len(center_vector) - 1):
            for j in range(i + 1, len(center_vector)):
                d_temp = abs(center_vector[i] - center_vector[j])
                if d_temp > d:
                    d = d_temp
    sigma = abs(d) / (2 * np.sqrt(2 * np.log(2)))
    while sigma < 1:
        sigma *= 10
    return sigma

def gaussmf(x, params):
    sigma, center = params
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def GaussMF(x1, label, MFnumber, sigma, centers):
    if 1 <= label <= MFnumber:
        return gaussmf(x1, [sigma, centers[int(label) - 1]])
    return 0


def RuleWeight(rules, data, cluster, center_vector,center_vector_label):
    data_num, attribute_num = data.shape
    sigma = np.zeros(attribute_num)
    t = np.zeros((data_num, attribute_num))
    for feature_index in range(attribute_num):
        feature_data = data[:, feature_index]
        rule_index = rules[:, feature_index]
        mf_number = cluster[feature_index]
        sigma[feature_index] = compute_sigma(center_vector[feature_index])
        
        for i in range(data_num):
            if(feature_index!=attribute_num-1):
                t[i, feature_index] = GaussMF(feature_data[i], rule_index[i], mf_number, sigma[feature_index], center_vector[feature_index])
            else:
                t[i, feature_index] = GaussMF(feature_data[i], rule_index[i], mf_number, sigma[feature_index], center_vector_label)
    return t, sigma
