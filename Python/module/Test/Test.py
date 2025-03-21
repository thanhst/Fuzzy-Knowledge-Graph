import numpy as np
from module.Membership_Function.GaussMF import GaussMF
from models.load_model import load_model
from module.Test.fuzzify_input import fuzzify_input
from module.Test.match_rule import match_rule

def test_fis(input_data,fileName):
    model_data = load_model(fileName = fileName)
    ruleList = np.array(model_data["ruleList"])
    sigma_M = np.array(model_data["sigma_M"]).flatten()
    centers = np.array(model_data["centers"], dtype=object)

    fuzzy_input = fuzzify_input(input_data, sigma_M, centers)
    predicted_label = match_rule(fuzzy_input, ruleList)
    if predicted_label is not None:
        return predicted_label
    else:
        return 0
