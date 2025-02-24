import numpy as np
def standardize_data_columnwise(data):
    data = np.array(data, dtype=np.float64)
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    
    std_vals[std_vals == 0] = 1
    
    standardized_data = (data - mean_vals) / std_vals
    return standardized_data, mean_vals, std_vals