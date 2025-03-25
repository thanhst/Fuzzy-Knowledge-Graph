import numpy as np
def gaussmf(x, params):
    sigma, center = params
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def GaussMF(x1, label, MFnumber, sigma, centers):
    if 1 <= label <= MFnumber:
        return gaussmf(x1, [sigma, centers[int(label) - 1]])
    return 0
