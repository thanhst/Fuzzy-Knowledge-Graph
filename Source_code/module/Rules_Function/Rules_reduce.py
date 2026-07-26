import numpy as np


def reduce_rule(h,col_num,rules):
    rules = np.asarray(rules)
    if rules.size == 0:
        return np.empty((0, col_num + 1))

    # FKG consumes sample-level rules. Do not collapse duplicate antecedents:
    # A->B, A->C, and repeated A->B rows all carry frequency/conflict evidence.
    return rules[:, :col_num + 1]

def remove_rule(h,col_num,rules):
    rules = np.asarray(rules)
    if rules.size == 0:
        return np.empty((0, col_num + 1))

    # Keep duplicates and conflicting consequents. The membership/repetition
    # signal is handled downstream by duplicate-aware rule matching/FKG counts.
    return rules[:, :col_num + 1]


def reduce_rule_unique_best(h, col_num, rules):
    """Legacy reducer: one consequent per antecedent, selected by certainty."""
    rule_dict = {}
    for rule in rules:
        condition = tuple(rule[:-3])
        value = rule[-2]
        label = rule[-3]
        result = [value, label]
        if condition in rule_dict:
            if rule_dict[condition][0] > result[0]:
                rule_dict[condition] = result
        else:
            rule_dict[condition] = result
    return np.array([[*key, value[1]] for key, value in rule_dict.items()])


def remove_rule_unique_high_confidence(h, col_num, rules, threshold=0.9):
    """Legacy filter: unique rules above a confidence threshold."""
    unique_rules = []
    for i in range(rules.shape[0]):
        if rules[i, rules.shape[1] - 1] >= threshold:
            unique_rules.append(tuple(rules[i]))
    if not unique_rules:
        return np.empty((0, col_num + 1))
    return np.array(list(set(unique_rules)))[:, :col_num + 1]
