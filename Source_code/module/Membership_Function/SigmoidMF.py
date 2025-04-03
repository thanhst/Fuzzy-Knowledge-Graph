# ==================== Sigmoid Membership Function ====================
import numpy as np
def sigmoid_mf(x, params):
    k, c = params
    return 1 / (1 + np.exp(-k * (x - c)))

def SigmoidMF(x1, label, MFnumber, sigma, centers):
    if 1 <= label <= MFnumber:
        c = centers[int(label) - 1]
        k = 1 / sigma
        return sigmoid_mf(x1, [k, c])
    return 0