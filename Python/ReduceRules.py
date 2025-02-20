import numpy as np

def calculate_D_Similarity(train_data_muy, train_data_alpha, row_muy, row_alpha):
    # Assuming this is a placeholder for the similarity calculation function
    # Implement this function as per your needs.
    return np.linalg.norm(train_data_muy - row_muy, axis=1) + np.linalg.norm(train_data_alpha - row_alpha, axis=1)

def reduce_rules(train_data_muy, train_data_alpha):
    test_num = train_data_muy.shape[0]
    threshold = 0.2
    index = 2  # Initiate value to pass the condition of the WHILE Loop
    train_data_muy_updated = np.copy(train_data_muy)
    train_data_alpha_updated = np.copy(train_data_alpha)
    train_data_index_updated = np.arange(test_num)
    
    cluster_index = 0
    candidates_muy = []
    candidates_alpha = []
    member_count = 1
    rules_index = 1
    rules_index_reduced = []

    while index > 1 and member_count > 0:
        index = 1
        member_count = 0
        D_Sim_temp = calculate_D_Similarity(train_data_muy_updated, train_data_alpha_updated, train_data_muy_updated[0, :], train_data_alpha_updated[0, :])
        
        test_num_update = train_data_muy_updated.shape[0]
        attribute_num_update = train_data_muy_updated.shape[1]
        
        train_data_muy_updated_temp = []
        train_data_alpha_updated_temp = []
        rules_index_updated = []
        
        if test_num_update > 1:
            for i in range(test_num_update):
                if D_Sim_temp[i] < threshold:
                    train_data_muy_updated_temp.append(train_data_muy_updated[i, :])
                    train_data_alpha_updated_temp.append(train_data_alpha_updated[i, :])
                    rules_index_updated.append(train_data_index_updated[i])
                    index += 1
                else:
                    if member_count < 1:
                        member_count += 1
                        candidates_muy.append(train_data_muy_updated[i, :])
                        candidates_alpha.append(train_data_alpha_updated[i, :])
                        rules_index = train_data_index_updated[i]
                        rules_index_reduced.append(rules_index)
                        cluster_index += 1

            if member_count < 1:
                candidates_muy.append(train_data_muy_updated_temp[0])
                candidates_alpha.append(train_data_alpha_updated_temp[0])
                train_data_muy_updated_temp = train_data_muy_updated_temp[1:]
                train_data_alpha_updated_temp = train_data_alpha_updated_temp[1:]
                rules_index = rules_index_updated[0]
                rules_index_reduced.append(rules_index)
                cluster_index += 1
                rules_index_updated = rules_index_updated[1:]
                member_count += 1

        if len(train_data_muy_updated_temp) > 1:
            train_data_muy_updated = np.array(train_data_muy_updated_temp)
            train_data_alpha_updated = np.array(train_data_alpha_updated_temp)
            train_data_index_updated[:index-2] = np.array(rules_index_updated)
        else:
            candidates_muy.append(train_data_muy_updated_temp[0])
            candidates_alpha.append(train_data_alpha_updated_temp[0])
            rules_index = rules_index_updated[0]
            rules_index_reduced.append(rules_index)
            member_count = 0

    return np.array(rules_index_reduced)
