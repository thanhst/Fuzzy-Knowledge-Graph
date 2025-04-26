import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def run_two_way_anova(data: pd.DataFrame):
    """
    Thực hiện ANOVA hai chiều dựa trên dataframe có 3 cột chính:
    - 'value': giá trị độ đo
    - 'method': tên phương pháp
    - 'metric': loại độ đo (accuracy, f1_score, train_time, test_time, ...)
    """
    # Kiểm tra đầu vào
    assert set(['value', 'method', 'metric']).issubset(data.columns), "Thiếu cột cần thiết"
    
    # Xây dựng mô hình ANOVA 2 chiều với tương tác
    model = ols('value ~ C(method) + C(metric) + C(method):C(metric)', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table
# Giả lập dữ liệu
data = {
    'value': [
        75.406, 76.479, 74.356, 75.578 , 77.106,  # accuracy - Filter
        74.26, 74.186, 73.741,73.244 ,74.5, # accuracy - Feature selection
        74.844,75.6, 74.426, 74.49, 73.904, # accuracy - Hadamard
        68.304, 68.498,67.204,68.958,65.328,  # accuracy - Tensor
        77.28, 77.526, 78.256,76.644,75.292,  # accuracy - Wrapper
        76.79, 78.315, 74.74,76.58,77.5 , # precision - Filter
        73.795, 75.039,74.23,73.990,76.289,  # precision - Feature selection
        76.110, 76.93, 76.585,78.025, 75.794, # precision - Hadamard
        68.07, 69.935, 65.91,65.97,62.86,  # precision - Tensor
        78.375, 72.875, 76.82,76.17,75.425,  # precision - Wrapper
        64.305, 65.86, 63.96, 64.664, 66.345, # recall - Filter
        64.395, 65.81, 64.1, 63.155,64.495, # recall - Feature selection
        61.8, 62.015, 61.61, 63.88, 60.315,  # recall - Hadamard
        61.58, 62.02,59.22,60.005,57.585,  # recall - Tensor
        65.555,62.12, 61.955,61.945,61.45,  # recall - Wrapper
        18.57, 18.899, 18.67,  18.700, 19.02 , # train time - Filter
        15.79, 15.59, 15.17,15.187 ,  15.43, # train time - Feature selection
        21.34, 21.202, 21.36, 18.478,19.383 ,  # train time - Hadamard
        8.255, 8.447, 8.210,8.208, 8.034,  # train time - Tensor
        1.533, 1.505, 1.421,1.439,1.470,  # train time - Wrapper
        178.05,177.999, 182.041,191.87,186.608,  # test time - Filter
        156.90, 159.466, 152.57,151.285 ,151.04, # test time - Feature selection
        261.61, 269.53, 279.288,207.473,241.549,  # test time - Hadamard
        59.367,60.306, 60.1133,59.864, 58.869, # test time - Tensor
        18.257, 17.204, 16.989,17.41,17.19, # test time - Wrapper
    ],
    'method': (
    ['Filter multimodal selection'] * 5 + ['Feature selection'] * 5 + ['Hadamard selection'] * 5 + ['Tensor selection'] * 5 + ['Wrapper-based Multimodal selection'] * 5 +
    ['Filter multimodal selection'] * 5 + ['Feature selection'] * 5 + ['Hadamard selection'] * 5 + ['Tensor selection'] * 5 + ['Wrapper-based Multimodal selection'] * 5 +
    ['Filter multimodal selection'] * 5 + ['Feature selection'] * 5 + ['Hadamard selection'] * 5 + ['Tensor selection'] * 5 + ['Wrapper-based Multimodal selection'] * 5 +
    ['Filter multimodal selection'] * 5 + ['Feature selection'] * 5 + ['Hadamard selection'] * 5 + ['Tensor selection'] * 5 + ['Wrapper-based Multimodal selection'] * 5 +
    ['Filter multimodal selection'] * 5 + ['Feature selection'] * 5 + ['Hadamard selection'] * 5 + ['Tensor selection'] * 5 + ['Wrapper-based Multimodal selection'] * 5
    ),
    'metric': (
        ['accuracy'] * 25 +
        ['precision'] * 25 +
        ['recall'] * 25 +
        ['train_time'] * 25 +
        ['test_time'] * 25
    )
}

df = pd.DataFrame(data)

# Chạy ANOVA
anova_result = run_two_way_anova(df)
print(anova_result)

tukey = pairwise_tukeyhsd(df['value'], df['method'], alpha=0.05)
print(tukey)
