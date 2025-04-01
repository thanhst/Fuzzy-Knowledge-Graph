import numpy as np

def fcm_function(X, C, V_init, m=2, eps=1e-5, max_iter=100):
    X = X.reshape(-1, 1)
    N, d = X.shape

    U = np.zeros((C, N))
    V = np.array(V_init, dtype=np.float64).reshape(-1, 1)

    J_prev = np.inf

    for count in range(max_iter):
        dist = np.zeros((N, C))

        for i in range(N):
            for j in range(C):
                diff = X[i, :] - V[j, :]
                dist[i, j] = np.sqrt(np.sum(diff ** 2))
                if dist[i, j] == 0:
                    dist[i, j] = np.finfo(float).eps

        U_new = np.zeros((C, N))
        for i in range(N):
            for j in range(C):
                sum_term = 0.0
                for k in range(C):
                    ratio = (dist[i, j] / dist[i, k]) ** (2 / (m - 1))
                    sum_term += ratio
                U_new[j, i] = 1.0 / sum_term
        U_new = np.nan_to_num(U_new, nan=0.0)

        U_new /= np.sum(U_new, axis=0)

        V_new = np.zeros((C, d))
        for j in range(C):
            for dim in range(d):
                numerator = np.sum((U_new[j, :] ** m) * X[:, dim])
                denominator = np.sum(U_new[j, :] ** m) + np.finfo(float).eps
                V_new[j, dim] = numerator / denominator

        J = 0.0
        for i in range(N):
            for j in range(C):
                J += (U_new[j, i] ** m) * (dist[i, j] ** 2)

        if np.abs(J - J_prev) < eps:
            break

        J_prev = J
        U = U_new.copy()
        V = V_new.copy()

    return V, U
