import numpy as np
def match_rule(fuzzy_input, ruleList):
    for rule in ruleList:
        if np.array_equal(fuzzy_input, rule[:-1]):
            return rule[-1]
    return None