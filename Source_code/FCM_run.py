from module.FCM.FCM_Function import fcm_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('./data/Dataset/OnlyImageFeture.csv')
train_data = train_data.values
cluster = [3,7,3,5,3,5,3,5,3,5,6]
h = train_data.shape[0]
w = train_data.shape[1]
min_vals = np.min(train_data, axis=0)
max_vals = np.max(train_data, axis=0)
U_all = []
centers = []
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
        
    center, U = fcm_function(feature, cluster[i], V, 2, 0.01, 100)
    U_list = U.T
    U_all.append(U_list)
    labels = np.argmax(U_list, axis=1)

    unique_labels, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(unique_labels, counts, color='skyblue', edgecolor='black')
    plt.xticks(unique_labels)
    plt.xlabel("Cluster Labels")
    plt.ylabel("Number of Data Points")
    plt.title("Total Number of Points in Each Cluster")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    centers.append(center.flatten())
    
    


