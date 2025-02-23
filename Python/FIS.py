import numpy as np
import pandas as pd
from FCM_Function import fcm_function as FCM_Function
from RuleWeight import RuleWeight
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/Result_norminal.csv')

# full_data = df.drop(df.columns[0], axis=1)

def standardize_data_columnwise(data):
    data = np.array(data, dtype=np.float64)
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    
    std_vals[std_vals == 0] = 1
    
    standardized_data = (data - mean_vals) / std_vals
    return standardized_data, mean_vals, std_vals

# labelR = df.iloc[:, 0].values.reshape(-1, 1)
full_data = df
# full_data = df.drop(df.columns[0], axis=1)

# full_data,mean_vals,std_vals = standardize_data_columnwise(full_data)

# full_data = np.hstack((full_data, labelR))


df_full_data = pd.DataFrame(full_data)
train_data = df_full_data.sample(frac=0.8, random_state=42)
train_data = train_data.values


full_data = np.array(full_data)
train_data = np.array(train_data)

min_vals = np.min(full_data, axis=0)
max_vals = np.max(full_data, axis=0)

min_vals_data = pd.DataFrame(min_vals)
max_vals_data = pd.DataFrame(max_vals)
min_vals_data.to_csv("./data/min_vals.csv")
max_vals_data.to_csv("./data/max_vals.csv")


h = train_data.shape[0]
w = train_data.shape[1]
cluster = [5, 5, 5, 5, 5, 5, 6]
m = 2
esp = 0.01
maxTest = 200
centers = []
rules = np.zeros((h, w))

# sns.pairplot(df)
# plt.show()

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
    for j in range(h):
        rules[j, i] = np.argmax(U[j, :]) + 1

col_num = train_data.shape[1] -1
label = train_data[:, col_num]

# seg = (max_vals[w-1] - min_vals[w-1]) / (cluster[w-1]-1)
# V = [min_vals[w-1]]
# for j in range(1, cluster[col_num]-1):
#     V.append(min_vals[col_num] + j * seg)
# V.append(max_vals[col_num])
# center, U = FCM_Function(feature, cluster[col_num], V, m, esp, maxTest)
# U = U.T
# for j in range(h):
#     rules[j, col_num] = np.argmax(U[j, :]) + 1
# center_vector_label = center.flatten()

for j in range(h):
    rules[j, col_num] = np.argmax(U[j, :]) + 1


[t, sigma_M] = RuleWeight(rules, train_data[:,:-1], cluster, centers)
sigma_M = sigma_M.reshape(-1,1)
sigma_M = sigma_M[:-1, :]

# Process sigma_M
sigma_M = np.hstack((sigma_M[:, [0]], sigma_M[:, [0]], sigma_M[:, [0]]))

rules = np.hstack((rules, np.min(t, axis=1, keepdims=True), train_data[:, [col_num]]))
unique_rules = []

df_Rule_List = pd.DataFrame(rules)
df_Rule_List.to_csv("./data/Rule_List_All.csv", index=False)

for i in range(h):
        for j in range(i + 1, h):
            if np.array_equal(rules[i, :col_num-2], rules[j, :col_num-2]):
                if rules[i, col_num-2] > rules[j, col_num-2]:
                    rules[j, :] = 0
                else:
                    rules[i, :] = 0
                    
for i in range(h):
    if int(rules[i, rules.shape[1]-1]) >= 0.9:
        unique_rules.append(tuple(rules[i]))
ruleList = np.array(list(set(unique_rules)))[:, :col_num+1]

df_Rule_List = pd.DataFrame(ruleList)
df_Rule_List.to_csv("./data/Rule_List.csv", index=False)

df_Sigma = pd.DataFrame(sigma_M)
df_Sigma.to_csv("./data/Sigma_M.csv", index=False)

df_Centers = pd.DataFrame(centers)
df_Centers.to_csv("./data/Centers.csv", index=False)

print("ruleList:",ruleList)
print("sigma_M:", sigma_M)
print("centers:", centers)

# correct_count = 0
# total_matched = 0  

# for rule in ruleList:
#     predicted_label = rule[col_num]  

#     idx = np.where((rules[:, :col_num] == rule[:col_num]).all(axis=1))[0]

#     if len(idx) > 0:
#         total_matched += len(idx)
#         actual_labels = train_data[idx, col_num]
#         correct_count += np.sum(actual_labels == predicted_label) 

# accuracy = correct_count / total_matched if total_matched > 0 else 0
# print(f"Độ chính xác của luật: {accuracy * 100:.2f}%")

