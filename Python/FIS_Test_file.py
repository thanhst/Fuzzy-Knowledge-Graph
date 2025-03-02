import numpy as np
from module.Membership_Function.GaussMF import GaussMF
from model.load_model import load_model
from module.Test.fuzzify_input import fuzzify_input
from module.Test.match_rule import match_rule
import pandas as pd
from module.Test.Test import test_fis
from sklearn.metrics import f1_score,recall_score,accuracy_score, confusion_matrix

predict_labels = []
true_labels = []

data = pd.read_csv("../Python/data/test_data.csv")
for i,r in data.iterrows():
    true_labels.append(r.values[7])
    sample_input = r.values[1:7]
    predict_labels.append(test_fis(sample_input))
true_labels = np.array(true_labels)
predicted_labels = np.array(predict_labels)
#Measurement
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Độ chính xác: {accuracy:.2%}")
f1 = f1_score(true_labels, predicted_labels,average='macro')
print(f"Độ đo F1: {f1:.2%}")
recall = recall_score(true_labels, predicted_labels,average='macro')
print(f"Độ đo Recall: {recall:.2%}")
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion matrix:\n")
print(conf_matrix)
