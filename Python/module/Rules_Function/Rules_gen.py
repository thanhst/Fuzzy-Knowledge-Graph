import numpy as np
from module.FCM.FCM_Function import fcm_function as FCM_Function
def rule_generate(h,w,train_data,cluster,min_vals,max_vals,m=2,esp=1e-5,maxTest=200,threshold =0.5):
    centers = []
    rules = np.zeros((h, w))
    for i in range(w):
        feature = train_data[:, i]
        if(cluster[i]==3):
            V = np.array([min_vals[i], (min_vals[i] + max_vals[i]) / 2, max_vals[i]])
        elif(cluster[i]==2):
            V = np.array([min_vals[i], max_vals[i]])
        elif(cluster[i]==1):
            V = np.array([(min_vals[i] + max_vals[i]) / 2])
        else:
            seg = (max_vals[i] - min_vals[i]) / (cluster[i]-1)
            V = [min_vals[i]]

            for j in range(1, cluster[i]-1):
                V.append(min_vals[i] + j * seg)
            V.append(max_vals[i])
            
        center, U = FCM_Function(feature, cluster[i], V, m, esp, maxTest)
        U = U.T
        centers.append(center.flatten())
        # for j in range(h):
        #     rules[j, i] = [idx + 1 for idx in range(cluster[i]) if U[j, idx] > threshold]
        for j in range(h):
            rules[j, i] = np.argmax(U[j, :]) + 1
    return rules,centers,U