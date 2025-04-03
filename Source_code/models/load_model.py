import numpy as np
import pickle
from module.Membership_Function.GaussMF import GaussMF
import os

path = os.getcwd()
def load_model(fileName = None):
    model_path = os.path.join(path,f"models/{fileName}/fuzzy_model.pkl")
    with open(model_path, "rb") as file:
        model_data = pickle.load(file)
    return model_data