import numpy as np
def match_rule(fuzzy_input, ruleList):
    matched_labels = []
    for rule in ruleList:
        if np.array_equal(fuzzy_input, rule[:-1]):
            matched_labels.append(rule[-1])
    if not matched_labels:
        return None

    labels, counts = np.unique(np.asarray(matched_labels), return_counts=True)
    max_count = np.max(counts)
    winners = set(labels[counts == max_count])
    for label in matched_labels:
        if label in winners:
            return label
    return matched_labels[0]
