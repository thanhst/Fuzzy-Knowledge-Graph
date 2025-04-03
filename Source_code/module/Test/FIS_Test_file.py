import numpy as np
from module.Membership_Function.GaussMF import GaussMF
from models.load_model import load_model
from module.Test.fuzzify_input import fuzzify_input
from module.Test.match_rule import match_rule
import pandas as pd
from module.Test.Test import test_fis
from sklearn.metrics import precision_score,f1_score,recall_score,accuracy_score, confusion_matrix
import time
import csv,os

def FIS_Test_file(Modality=None,Turn =None,fileName=None):
    # base_dir = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.getcwd()
    startTime = time.time()
    predict_labels = []
    true_labels = []
    rules =[]
    
    data = pd.read_csv(os.path.join(base_dir,f"data/FIS/input/{fileName}/test_data.csv"))
    # print(os.path.join(base_dir,f"data/FIS/input/{fileName}/test_data.csv"))
    label_index = data.shape[1]-1
    for i,r in data.iterrows():
        true_labels.append(r.values[label_index])
        sample_input = r.values[1:label_index]
        label,rule = test_fis(sample_input,fileName)
        predict_labels.append(label)
        rule = np.append(rule, r.values[label_index])
        rules.append(rule)
    # df_rule_test = pd.DataFrame(rules)
    # df_rule_test.to_csv(os.path.join(base_dir,f"data/FIS/output/{fileName}/TestDataRule.csv"),index=False)
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predict_labels)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Độ chính xác: {accuracy:.2%}")
    precision = precision_score(true_labels, predicted_labels,average='macro')
    print(f"Độ đo precision: {precision:.2%}")
    f1 = f1_score(true_labels, predicted_labels,average='macro')
    print(f"Độ đo F1: {f1:.2%}")
    recall = recall_score(true_labels, predicted_labels,average='macro')
    print(f"Độ đo Recall: {recall:.2%}")
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion matrix:\n")
    print(conf_matrix)
    


    
    csv_file = os.path.join(base_dir,f"data/Test/{fileName}/acc.csv")
    if(Turn!=None):
        with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["id","Modality","Model","Accuracy","Precision Score","F1 Score", "Recall"])

            writer.writerow([Turn,Modality,"FIS",f"{accuracy:.2%}", f"{precision:.2%}",f"{f1:.2%}", f"{recall:.2%}"])
    endTime = time.time() - startTime
    print("Test xong : ", endTime)
    
    results = {
        "Total Time": [endTime],
        "Test Accuracy": [accuracy],
        "Test Precision": [precision],
        "Test Recall": [recall],
        "Test F1": [f1],
    }
    dfData = pd.DataFrame(results)
    dfData.to_csv(os.path.join(base_dir,f"data/FIS/output/{fileName}/Results_FIS.csv"), index=False)
