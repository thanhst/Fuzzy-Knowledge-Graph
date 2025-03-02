import numpy as np
import pickle
from module.Membership_Function.GaussMF import GaussMF

def load_model(model_path="../Python/model/fuzzy_model.pkl"):
    with open(model_path, "rb") as file:
        model_data = pickle.load(file)
    return model_data