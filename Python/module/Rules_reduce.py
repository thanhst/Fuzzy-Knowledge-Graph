import numpy as np
def reduce_rule(h,col_num,rules):
    for i in range(h):
        for j in range(i + 1, h):
            if np.array_equal(rules[i, :col_num-2], rules[j, :col_num-2]):
                if rules[i, col_num-2] > rules[j, col_num-2]:
                    rules[j, :] = 0
                else:
                    rules[i, :] = 0
    return rules

def remove_rule(h,col_num,rules):
    unique_rules=[]
    for i in range(h):
        if int(rules[i, rules.shape[1]-1]) >= 0.9:
            unique_rules.append(tuple(rules[i]))
    ruleList = np.array(list(set(unique_rules)))[:, :col_num+1]
    return ruleList