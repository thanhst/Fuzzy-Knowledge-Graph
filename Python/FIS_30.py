def FIS_30(Turn = None):
    import numpy as np
    import pandas as pd
    from module.Rules_Function.RuleWeight import RuleWeight
    import seaborn as sns
    import matplotlib.pyplot as plt
    from module.Rules_Function.Rules_gen import rule_generate
    from module.Rules_Function.Rules_reduce import reduce_rule,remove_rule
    import pickle
    from module.Convert.var_lang import change_var_lang̣
    import sys
    import time

    start_time = time.time()

    df = pd.read_csv('./data/Meta_result_remove_30_txl.csv')

    # full_data = df.drop(df.columns[0], axis=1

    # labelR = df.iloc[:, 0].values.reshape(-1, 1)
    full_data = df
    # full_data = df.drop(df.columns[0], axis=1)

    # full_data,mean_vals,std_vals = standardize_data_columnwise(full_data)

    # full_data = np.hstack((full_data, labelR))


    df_full_data = pd.DataFrame(full_data)
    train_data = df_full_data.sample(frac=0.8, random_state=None)
    train_data.to_csv("./data/FIS/input/train_data.csv")
    train_data = train_data.values
    test_data = df_full_data.sample(frac=0.2, random_state=None)
    print(train_data.shape)


    full_data = np.array(full_data)
    train_data = np.array(train_data)

    min_vals = np.min(full_data, axis=0)
    max_vals = np.max(full_data, axis=0)

    min_vals_data = pd.DataFrame(min_vals)
    max_vals_data = pd.DataFrame(max_vals)
    min_vals_data.to_csv("./data/FIS/output/min_vals.csv")
    max_vals_data.to_csv("./data/FIS/output/max_vals.csv")
    test_data.to_csv("./data/FIS/input/test_data.csv")

    h = train_data.shape[0]
    w = train_data.shape[1]

    cluster = [3,3,3,3,3,3,3,3,3,5,5,5,5,5,5,6]
    # cluster = [5,5,5,5,5,5,6]
    # print(len(cluster))

    lang2 = ["Low","High"]
    lang3 = ["Low","Medium","High"]
    lang5 = ["Very Low","Low","Medium","High","Very High"]
    # cluster = [2,2,3 , 3, 3, 2,2,2,2,2,2,3,3,3,3,2,3,2,3,2,2,2,5,5,5,5,5,5, 6]
    lang_matrix = [lang3,lang3,lang3,lang3,lang3,lang3,lang3,lang3,lang3,lang5,lang5,lang5,lang5,lang5,lang5]
    # lang_matrix = [lang5,lang5,lang5,lang5,lang5,lang5]
    print(len(lang_matrix))

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
    df_Rule_List.to_csv("./data/FIS/output/Rule_List_All.csv", index=False)

    rules_reduce = reduce_rule(h,col_num,rules)
    df_Rule_List1 = pd.DataFrame(rules_reduce)
    df_Rule_List1.to_csv("./data/FIS/output/Rule_List_reduce.csv", index=False)

    ruleList = remove_rule(h,col_num,rules_reduce)
    ruleListLang = change_var_lang̣(lang_matrix,ruleList)
    # print(ruleListLang)
    df_rule_lang = pd.DataFrame(ruleListLang)
    df_rule_lang.to_csv("./data/FIS/output/Rule_List_Language.csv",index=False)

    df_Rule_List = pd.DataFrame(ruleList)
    df_Rule_List.to_csv("./data/FIS/output/Rule_List.csv", index=False)

    df_Sigma = pd.DataFrame(sigma_M)
    df_Sigma.to_csv("./data/FIS/output/Sigma_M.csv", index=False)

    df_Centers = pd.DataFrame(centers)
    df_Centers.to_csv("./data/FIS/output/Centers.csv", index=False)

    print("ruleList:",ruleList)
    print("sigma_M:", sigma_M)
    print("centers:", centers)
    model_data = {
        "ruleList": ruleList,
        "sigma_M": sigma_M,
        "centers": centers,
        "min_vals": min_vals,
        "max_vals": max_vals
    }
    totalTime = time.time() - start_time
    print("Train finish : ",totalTime)
    with open("./models/fuzzy_model.pkl", "wb") as file:
        pickle.dump(model_data, file)

    from module.FKG.FKG import FKG
    base = [[int(float(x)) for x in row] for row in ruleList]
    base = pd.DataFrame(base)
    fkg_instance = FKG()
    fkg_instance.FKG(df = base,Turn=Turn,Modality="Metadata-Image Fusion")
    from FIS_Test_file import FIS_Test_file
    FIS_Test_file(Modality = "Metadata-Image Fusion",Turn = Turn)
    totalTime = time.time() - start_time
    print("All finish : ",totalTime)
    # for rule in ruleList:
    #     predicted_label = rule[col_num]

    #     idx = np.where((rules[:, :col_num] == rule[:col_num]).all(axis=1))[0]

    #     if len(idx) > 0:
    #         total_matched += len(idx)
    #         actual_labels = train_data[idx, col_num]
    #         correct_count += np.sum(actual_labels == predicted_label) 

    # accuracy = correct_count / total_matched if total_matched > 0 else 0
    # print(f"Độ chính xác của luật: {accuracy * 100:.2f}%")



