import numpy as np

def fcm_function(X, C, V, m=2, eps=1e-5, maxTest=100):
    X = X.reshape(-1, 1)
    N, r = X.shape
    C= int(C)
    V = np.array(V).reshape(-1, 1)
    # print(V.shape,N,C)
    U = np.zeros((C, N))
    count = 0
    
    while True:
        for i in range(N):
            for j in range(C):
                Nominator = np.linalg.norm(X[i, :] - V[j, :])
                
                if Nominator == 0:
                    U[j, i] = 1
                    continue
                Sum = np.sum([
                    (Nominator / np.linalg.norm(X[i, :] - V[l, :])) ** (2 / (m - 1))
                    if np.linalg.norm(X[i, :] - V[l, :]) != 0 else 0
                    for l in range(C)
                ])

                U[j, i] = 1 / Sum if Sum != 0 else 1 
        
        # Normalize U
        max_U = np.max(U, axis=0)
        for i in range(N):
            if max_U[i] == 1:
                U[:, i] = (U[:, i] == max_U[i]).astype(int)
        
        # Cập nhật các trung tâm V
        W = np.zeros_like(V)
        for j in range(C):
            for i in range(r):
                W_nominator = np.sum((U[j,:] ** m) * X[:, i])
                W_denominator = np.sum(U[j,:] ** m)
                if W_denominator != 0:
                    W[j,i] = W_nominator / W_denominator
                else:
                    W[j,i] = 0
        
        # Kiểm tra độ thay đổi
        diff = np.linalg.norm(W - V)
        
        if diff <= eps or count >= maxTest:
            break
        V = W.copy()
        count += 1
    
    # Tính toán giá trị hàm mục tiêu J
    J = 0
    for i in range(N):
        for j in range(C):
            sum2 = np.linalg.norm(X[i, :] - V[j, :]) ** 2
            J += (U[j, i] ** m) * sum2

    # Xác định cụm của từng mẫu
    cum = np.argmax(U, axis=0)
    
    return V, U
