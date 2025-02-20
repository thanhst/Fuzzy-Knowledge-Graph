import numpy as np
from FCM_Function import fcm_function as FCM
from scipy.io import savemat
import pandas as pd

def Make(train_data):
    train_data = pd.read_csv(train_data)
    data_num, attribute_num = train_data.shape
    data = train_data[:, :-1]  # All columns except the last one (features)
    nhan1 = train_data[:, -1]  # Last column (labels)

    cluster = np.zeros(attribute_num)  # Initialize cluster type for each feature

    # Determine the number of clusters based on unique values in each feature
    for i in range(attribute_num):
        temp = len(np.unique(train_data[:, i]))
        if temp == 2:
            cluster[i] = 2
        elif temp == 3:
            cluster[i] = 3
        else:
            cluster[i] = 3

    m = 2
    esp = 0.01
    maxTest = 200
    center_vector = [None] * attribute_num
    centers = [None] * (attribute_num - 1)

    X1 = np.zeros((data_num, attribute_num))

    # Perform Fuzzy C-Means Clustering for each feature
    for feature_index in range(attribute_num):
        feature_data = train_data[:, feature_index]
        V = np.zeros((3, 1))  # Initialize cluster centers

        min_value = np.min(train_data[:, feature_index])
        max_value = np.max(train_data[:, feature_index])
        delta = max_value - min_value
        
        if cluster[feature_index] == 2:
            V[0, 0] = min_value - 0.5
            V[1, 0] = max_value
        else:
            V[0, 0] = min_value
            V[1, 0] = min_value + delta / 2
            V[2, 0] = max_value

        # Perform Fuzzy C-Means Clustering
        fcm = FCM(n_clusters=int(cluster[feature_index]))
        fcm.fit(feature_data.reshape(-1, 1))  # Reshape to 2D array for FCM
        U = fcm.u.T  # Transpose membership matrix
        center = fcm.centers

        center_vector[feature_index] = center[:, 0]

        # Assign clusters based on the maximum membership value
        for i in range(data_num):
            maximum = np.max(U[i, :])
            for j in range(int(cluster[feature_index])):
                if maximum == U[i, j]:
                    X1[i, feature_index] = j + 1  # +1 to match 1-based indexing

    # Label calculation (same as in the original MATLAB code)
    label = train_data[:, 0] + 1  # Labels should start from 1

    # Prepare the ruleList1 matrix
    ruleList1 = X1.astype(int)

    # Calculate the A matrix (relationship matrix)
    r1, r2 = ruleList1.shape
    A = np.zeros((r1, int((r2 - 3) * (r2 - 2) * (r2 - 1) / 6)))  # A matrix size calculation
    j = 0
    for i in range(r1):
        for l in range(r2 - 3):
            for h in range(l + 1, r2 - 2):
                for s in range(h + 1, r2 - 1):
                    count = 0
                    for t in range(r1):
                        if (ruleList1[t, l] == ruleList1[i, l] and
                                ruleList1[t, h] == ruleList1[i, h] and
                                ruleList1[t, s] == ruleList1[i, s]):
                            count += 1
                    A[i, j] = count / r1
                    j += 1

    # Calculate the B matrix (another relationship matrix)
    B = np.zeros_like(A)
    T = np.sum(A, axis=1)
    for i in range(r1):
        j = 0
        for l in range(r2 - 2):
            for h in range(l + 1, r2 - 1):
                dem1 = 0
                dem2 = 0
                for t in range(r1):
                    if (ruleList1[t, l] == ruleList1[i, l] and ruleList1[t, r2 - 1] == ruleList1[i, r2 - 1]):
                        dem1 += 1
                    if (ruleList1[t, h] == ruleList1[i, h] and ruleList1[t, r2 - 1] == ruleList1[i, r2 - 1]):
                        dem2 += 1
                M1 = dem1 / r1
                M2 = dem2 / r1
                B[i, j] = T[i] * min(M1, M2)
                j += 1

    # Save results to a .mat file
    result_dict = {
        'X1': X1,
        'label': label,
        'A': A,
        'B': B,
        'ruleList1': ruleList1
    }

    savemat('output.mat', result_dict)

    return X1, A, B, center_vector

# Example usage:
# train_data is assumed to be a numpy array or a pandas DataFrame with your training data.
  # Example random data
X1, A, B, center_vector = Make('Tan1.txt')
