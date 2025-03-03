import numpy as np
def change_var_lang̣(matrix_lang,data):
    result = np.empty(data.shape, dtype=object)
    for i in range(data.shape[0]):
        for j in range(data.shape[-1]-1):
            result[i, j] = matrix_lang[j][int(data[i, j]) - 1]
        result[i,-1] = data[i,-1]
    return result