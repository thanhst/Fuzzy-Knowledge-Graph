# ==================== Exp Membership Function ====================
import numpy as np

def exp_mf(x, params):
    sigma, center = params
    lambda_ = 1 / sigma
    return np.exp(-lambda_ * abs(x - center))

def ExpMF(x1, label, MFnumber, sigma, centers):
    if 1 <= label <= MFnumber:
        return exp_mf(x1, [sigma, centers[int(label) - 1]])
    return 0