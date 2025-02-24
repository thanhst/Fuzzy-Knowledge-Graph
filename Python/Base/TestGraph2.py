import numpy as np
import time

def TestGraph2(Test, A, B, X1, cluster, center_vector):
    eta = 0.000001
    Test_num, test_attribute_num = Test.shape
    datatest = Test[:, :-1]
    nhantest = Test[:, -1]

    start_time_train = time.time()

    # Initialize an empty array for cluster assignments
    X2 = np.zeros_like(datatest, dtype=int)

    # Calculate cluster assignment for each test point
    for j in range(test_attribute_num - 1):
        w = center_vector[j]
        for i in range(Test_num):
            d = np.ones(len(w)) * 1000  # Set all distances initially to a large value
            for l in range(len(w)):
                d[l] = datatest[i, j] - w[l]
            dmin = np.min(d)
            X2[i, j] = np.argmin(d)  # Find index of the minimum distance

    T = X2  # Use cluster assignment as T

    ruleList1 = X1
    te = T
    a1, a2 = te.shape
    r1, r2 = ruleList1.shape

    # Initialize matrices C1, C2, and C3 for storing results
    C1 = np.zeros((a1, a2 - 2))
    C2 = np.zeros((a1, a2 - 2))
    C3 = np.zeros((a1, a2 - 2))

    # Compute C1, C2, C3
    for i in range(a1):
        for l in range(a2 - 2):
            for h in range(l + 1, a2 - 1):
                dem1 = dem2 = dem3 = 0
                for t in range(r1):
                    if te[i, l] == ruleList1[t, l] and te[i, h] == ruleList1[t, h]:
                        if ruleList1[t, r2 - 1] == 1:
                            dem1 += B[t, 0]
                        elif ruleList1[t, r2 - 1] == 2:
                            dem2 += B[t, 0]
                        else:
                            dem3 += B[t, 0]
                C1[i, l] = dem1
                C2[i, l] = dem2
                C3[i, l] = dem3

    minC1 = np.min(C1, axis=1)
    maxC1 = np.max(C1, axis=1)
    D1 = minC1 + maxC1

    minC2 = np.min(C2, axis=1)
    maxC2 = np.max(C2, axis=1)
    D2 = minC2 + maxC2

    minC3 = np.min(C3, axis=1)
    maxC3 = np.max(C3, axis=1)
    D3 = minC3 + maxC3

    nhan = np.zeros(a1)

    # Classification based on distances D1, D2, D3
    for k in range(a1):
        if D1[k] > D2[k] and D1[k] > D3[k]:
            nhan[k] = 0
        elif D3[k] > D2[k] and D3[k] > D1[k]:
            nhan[k] = 1
        else:
            nhan[k] = 2

    end_time_train = time.time()
    Time = end_time_train - start_time_train

    # Compute accuracy
    temp = nhan
    train_output = nhantest
    dem = np.sum(temp == train_output)
    
    Accuracy = dem / len(temp)

    return Accuracy, Time
