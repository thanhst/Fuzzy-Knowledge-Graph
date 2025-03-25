import numpy as np
from module.Membership_Function.GaussMF import GaussMF

def fuzzify_input(input_data, sigma_M, centers):
    fuzzy_values = []
    for i in range(len(input_data)):
        center_vector = centers[i]
        sigma = sigma_M[i]
        membership_values = [GaussMF(input_data[i], label, len(center_vector), sigma, center_vector) for label in range(1, len(center_vector) + 1)]
        # print(f"Input: {input_data[i]}, Membership values: {membership_values}")
        fuzzy_values.append(np.argmax(membership_values) + 1)
    
    return np.array(fuzzy_values)
