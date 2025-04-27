import numpy as np
from module.Membership_Function.GaussMF import GaussMF
from models.load_model import load_model
from module.Test_FIS.fuzzify_input import fuzzify_input
from module.Test_FIS.match_rule import match_rule
from sklearn.model_selection import train_test_split
import pandas as pd
import os
path = os.getcwd()

def test_fis(input_data,fileName):
    model_data = load_model(fileName = fileName)
    ruleList = np.array(model_data["ruleList"])
    sigma_M = np.array(model_data["sigma_M"]).flatten()
    centers = np.array(model_data["centers"], dtype=object)
    
    # ruleList, _ = train_test_split(ruleList, test_size=0.3, random_state=None)

    fuzzy_input = fuzzify_input(input_data, sigma_M, centers)
    
    predicted_label = match_rule(fuzzy_input, ruleList)
    if predicted_label is not None:
        return predicted_label,fuzzy_input
    else:
        return 0,fuzzy_input
