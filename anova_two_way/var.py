import pandas as pd
import numpy as np

# Dữ liệu
data = {
    'value': [
        # Accuracy
        75.406, 76.479, 74.356, 75.578 , 77.106,
        74.26, 74.186, 73.741,73.244 ,74.5,
        74.844,75.6, 74.426, 74.49, 73.904,
        68.304, 68.498,67.204,68.958,65.328,
        77.28, 77.526, 78.256,76.644,75.292,

        # Precision
        76.79, 78.315, 74.74,76.58,77.5,
        73.795, 75.039,74.23,73.990,76.289,
        76.110, 76.93, 76.585,78.025, 75.794,
        68.07, 69.935, 65.91,65.97,62.86,
        78.375, 72.875, 76.82,76.17,75.425,

        # Recall
        64.305, 65.86, 63.96, 64.664, 66.345,
        64.395, 65.81, 64.1, 63.155,64.495,
        61.8, 62.015, 61.61, 63.88, 60.315,
        61.58, 62.02,59.22,60.005,57.585,
        65.555, 62.12, 61.955,61.945,61.45,

        # Train time
        18.57, 18.899, 18.67,  18.700, 19.02,
        15.79, 15.59, 15.17,15.187 ,  15.43,
        21.34, 21.202, 21.36, 18.478,19.383,
        8.255, 8.447, 8.210,8.208, 8.034,
        1.533, 1.505, 1.421,1.439,1.470,

        # Test time
        178.05,177.999, 182.041,191.87,186.608,
        156.90, 159.466, 152.57,151.285 ,151.04,
        261.61, 269.53, 279.288,207.473,241.549,
        59.367,60.306, 60.1133,59.864, 58.869,
        18.257, 17.204, 16.989,17.41,17.19,
    ]
}

metrics = ['accuracy', 'precision', 'recall', 'train time', 'test time']
methods = ['Filter', 'Feature selection', 'Hadamard', 'Tensor', 'Wrapper']
repeats = 5

# Gán nhãn
df = pd.DataFrame(data)
df['metric'] = np.repeat(metrics, len(methods) * repeats)
df['method'] = np.tile(methods, len(metrics) * repeats)

# Đổi chiều của dữ liệu để có mỗi hàng là phương pháp
values = np.array(df['value']).reshape(len(metrics), len(methods), repeats)

# Tính trung bình và độ lệch chuẩn theo phương pháp (theo hàng)
means = np.mean(values, axis=2)
stds = np.std(values, axis=2)
total_time_means = means[3] + means[4]
total_time_stds = np.sqrt(stds[3]**2 + stds[4]**2)
# Tạo dataframe kết quả
summary_df = pd.DataFrame({
    'Method': methods,
    'Accuracy': [f"{means[0][i]:.3f} ± {stds[0][i]:.3f}" for i in range(len(methods))],
    'Precision': [f"{means[1][i]:.3f} ± {stds[1][i]:.3f}" for i in range(len(methods))],
    'Recall': [f"{means[2][i]:.3f} ± {stds[2][i]:.3f}" for i in range(len(methods))],
    'Train time': [f"{means[3][i]:.3f} ± {stds[3][i]:.3f}" for i in range(len(methods))],
    'Test time': [f"{means[4][i]:.3f} ± {stds[4][i]:.3f}" for i in range(len(methods))],
    'Total time': [f"{total_time_means[i]:.3f} ± {total_time_stds[i]:.3f}" for i in range(len(methods))],
})

# Hiển thị kết quả
print(summary_df)
