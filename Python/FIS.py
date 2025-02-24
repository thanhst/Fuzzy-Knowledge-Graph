import numpy as np
import pandas as pd
from RuleWeight import RuleWeight
import seaborn as sns
import matplotlib.pyplot as plt
from module.Rules_gen import rule_generate
from module.Rules_reduce import reduce_rule,remove_rule
df = pd.read_csv('./data/Result_norminal.csv')

# full_data = df.drop(df.columns[0], axis=1

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
cluster = [5, 7, 5, 5, 5, 7, 6]
m = 2
esp = 0.01
maxTest = 200

rules,centers,U = rule_generate(h,w,train_data,cluster,min_vals,max_vals,m,esp,maxTest)

# sns.pairplot(df)
# plt.show()

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

rules_reduce = reduce_rule(h,col_num,rules)
ruleList = remove_rule(h,col_num,rules)

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

