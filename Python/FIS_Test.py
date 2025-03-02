import numpy as np
from module.Membership_Function.GaussMF import GaussMF
from model.load_model import load_model
from module.Test.fuzzify_input import fuzzify_input
from module.Test.match_rule import match_rule
import pandas as pd
from module.Test.Test import test_fis

sample_input = np.array([1.220541,0.67459,0.713568,0.372083,0.697448,0.138652])
print("Nhãn dự đoán là : ",test_fis(sample_input))

