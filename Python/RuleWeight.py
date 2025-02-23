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

# ==================== Exp Membership Function ====================


def exp_mf(x, params):
    sigma, center = params
    lambda_ = 1 / sigma
    return np.exp(-lambda_ * abs(x - center))

def ExpMF(x1, label, MFnumber, sigma, centers):
    if 1 <= label <= MFnumber:
        return exp_mf(x1, [sigma, centers[int(label) - 1]])
    return 0

# ==================== Triangular Membership Function ====================

def triangle_mf(x, params):
    a, b, c = params
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    return 0

def TriangleMF(x1, label, MFnumber, sigma, centers):
    if 1 <= label <= MFnumber:
        center = centers[int(label) - 1]
        a = center - sigma
        b = center
        c = center + sigma
        return triangle_mf(x1, [a, b, c])
    return 0

# ==================== Trapezoidal Membership Function ====================

def trapezoid_mf(x, params):
    a, b, c, d = params
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1
    elif c < x <= d:
        return (d - x) / (d - c)
    return 0

def TrapezoidMF(x1, label, MFnumber, sigma, centers):
    if 1 <= label <= MFnumber:
        center = centers[int(label) - 1]
        a = center - 2 * sigma
        b = center - sigma
        c = center + sigma
        d = center + 2 * sigma
        return trapezoid_mf(x1, [a, b, c, d])
    return 0


# ==================== Sigmoid Membership Function ====================

def sigmoid_mf(x, params):
    k, c = params
    return 1 / (1 + np.exp(-k * (x - c)))

def SigmoidMF(x1, label, MFnumber, sigma, centers):
    if 1 <= label <= MFnumber:
        c = centers[int(label) - 1]
        k = 1 / sigma
        return sigmoid_mf(x1, [k, c])
    return 0


def RuleWeight(rules, data, cluster, center_vector):
    data_num, attribute_num = data.shape
    sigma = np.zeros(attribute_num)
    t = np.zeros((data_num, attribute_num))
    for feature_index in range(attribute_num):
        feature_data = data[:, feature_index]
        rule_index = rules[:, feature_index]
        mf_number = cluster[feature_index]
        sigma[feature_index] = compute_sigma(center_vector[feature_index])
        
        for i in range(data_num):
                t[i, feature_index] = TriangleMF(feature_data[i], rule_index[i], mf_number, sigma[feature_index], center_vector[feature_index])
    return t, sigma
