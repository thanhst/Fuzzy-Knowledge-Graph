import numpy as np
import pandas as pd
from FCM_Function import fcm_function as FCM_Function
from RuleWeight import RuleWeight

df = pd.read_csv('Tan1.txt')

# full_data = df.drop(df.columns[0], axis=1)
# train_data = df.drop(df.columns[0], axis=1)
# train_data = df.drop(df.columns[0], axis=1)


labelR = df.iloc[:, 0].values.reshape(-1, 1)

full_data = df.drop(df.columns[0], axis=1)
full_data = np.hstack((full_data, labelR))

df_full_data = pd.DataFrame(full_data)
train_data = df_full_data.sample(frac=0.8, random_state=42)
train_data = train_data.values

full_data = np.array(full_data)
train_data = np.array(train_data)

min_vals = np.min(full_data, axis=0)
max_vals = np.max(full_data, axis=0)

h = train_data.shape[0]
w = train_data.shape[1]
cluster = [3, 3, 3, 3, 3, 3, 7]
m = 2
esp = 0.01
maxTest = 200
centers = np.zeros((6, 3))
rules = np.zeros((h, w))


for i in range(w-1):
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
    centers[i, :] = center.flatten()
    for j in range(h):
        rules[j, i] = np.argmax(U[j, :]) + 1

seg = (max_vals[w-1] - min_vals[w-1]) / (cluster[w-1]-1)
V = [min_vals[w-1]]
for j in range(1, len(cluster)-2):
    V.append(min_vals[i] + j * seg)
V.append(max_vals[i])
center, U = FCM_Function(feature, cluster[i], V, m, esp, maxTest)
U = U.T
for j in range(h):
    rules[j, i] = np.argmax(U[j, :]) + 1
center_vector_label = center.flatten()

col_num = train_data.shape[1] -1
label = train_data[:, col_num]
V = np.linspace(min_vals[col_num], max_vals[col_num], cluster[col_num])
center, U = FCM_Function(label, cluster[col_num], V, m, esp, maxTest)
U = U.T
for j in range(h):
    rules[j, col_num] = np.argmax(U[j, :]) + 1


[t, sigma_M] = RuleWeight(rules, train_data[:,:-1], cluster, centers,center_vector_label)
sigma_M = sigma_M.reshape(-1,1)
sigma_M = sigma_M[:-1, :]

# Process sigma_M
sigma_M = np.hstack((sigma_M[:, [0]], sigma_M[:, [0]], sigma_M[:, [0]]))

rules = np.hstack((rules, np.min(t, axis=1, keepdims=True), train_data[:, [col_num]]))
unique_rules = []
for i in range(h):
        for j in range(i + 1, h):
            if np.array_equal(rules[i, :col_num-2], rules[j, :col_num-2]):
                if rules[i, col_num-2] > rules[j, col_num-2]:
                    rules[j, :] = 0
                else:
                    rules[i, :] = 0
                    
for i in range(h):
    if int(rules[i, rules.shape[1]-1]) >= 0.3:
        unique_rules.append(tuple(rules[i]))
ruleList = np.array(list(set(unique_rules)))[:, :col_num+1]

df_Rule_List = pd.DataFrame(ruleList)
df_Rule_List.to_csv("Rule_List.csv", index=False)  # Lưu mà không có cột index

df_Sigma = pd.DataFrame(sigma_M)
df_Sigma.to_csv("Sigma_M.csv", index=False)

df_Centers = pd.DataFrame(centers)
df_Centers.to_csv("Centers.csv", index=False)

print("ruleList:",ruleList)
print("sigma_M:", sigma_M)
print("centers:", centers)
