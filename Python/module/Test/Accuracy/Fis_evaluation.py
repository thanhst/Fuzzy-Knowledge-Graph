import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

rule_list = pd.read_csv("./data/FIS/output/Rule_List.csv").values
test_data = pd.read_csv("./data/FIS/input/test_data.csv").values

col_num = test_data.shape[1] - 1
y_true = test_data[:, col_num]

y_pred = []
for sample in test_data:
    matched_rules = rule_list[(rule_list[:, :-2] == sample[:-1]).all(axis=1)]
    if len(matched_rules) > 0:
        y_pred.append(matched_rules[0, -1])
    else:
        y_pred.append(-1)

y_pred = np.array(y_pred)

valid_indices = y_pred != -1
y_true_valid = y_true[valid_indices]
y_pred_valid = y_pred[valid_indices]

accuracy = accuracy_score(y_true_valid, y_pred_valid)
precision = precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
recall = recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
f1 = f1_score(y_true_valid, y_pred_valid, average='macro')

evaluation_metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value": [accuracy, precision, recall, f1]
})
evaluation_metrics.to_csv("./data/FIS/output/Evaluation_Metrics.csv", index=False)

print(evaluation_metrics)
