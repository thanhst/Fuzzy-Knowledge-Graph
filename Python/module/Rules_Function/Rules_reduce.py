import numpy as np
def reduce_rule(h,col_num,rules):
    rule_dict = {}
    for rule in rules:
        condition = tuple(rule[:-3])
        value = rule[-2]
        label = rule[-3]
        result= [value,label]
        if condition in rule_dict:
            if(rule_dict[condition][0]> result[0]):
                rule_dict[condition] = result
        else:
            rule_dict[condition] = result
    reduced_rules = np.array([[*key, value[1]] for key, value in rule_dict.items()])
    return reduced_rules

def remove_rule(h,col_num,rules):
    unique_rules=[]
    for i in range(rules.shape[0]):
        if (rules[i, rules.shape[1]-1]) >= 0.9:
            unique_rules.append(tuple(rules[i]))
    ruleList = np.array(list(set(unique_rules)))[:, :col_num+1]
    return ruleList