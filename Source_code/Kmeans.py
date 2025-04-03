import numpy as np

def compute_A(X_relations, R_size):
    """
    Tính ma trận A từ mối quan hệ giữa các đặc trưng.
    X_relations: Ma trận biểu diễn số lần xuất hiện của cặp (Xi, Xj) trong mỗi quy tắc.
    R_size: Tổng số quy tắc.
    """
    return X_relations / R_size

def compute_B(A_matrix, label_relations, R_size):
    """
    Tính ma trận B dựa trên A_matrix và quan hệ giữa các đặc trưng với nhãn.
    """
    B_matrix = np.sum(A_matrix, axis=1)[:, np.newaxis] * (label_relations / R_size)
    return B_matrix

def compute_C(B_matrix):
    """
    Tính ma trận C bằng cách tổng hợp tất cả giá trị B theo các quy tắc t.
    """
    return np.sum(B_matrix, axis=0)

def compute_D(C_matrix):
    """
    Áp dụng phép toán MIN-MAX để tính D.
    """
    return np.min(C_matrix, axis=0) + np.max(C_matrix, axis=0)

def classify(D_matrix):
    """
    Xác định nhãn cuối cùng bằng cách chọn nhãn có giá trị cao nhất trong D.
    """
    return np.argmax(D_matrix)

# Ví dụ dữ liệu đầu vào (giả định có 3 đặc trưng, 2 nhãn, 4 quy tắc)
X_relations = np.array([
    [3, 2, 1],
    [2, 3, 1],
    [1, 2, 3],
    [3, 1, 2]
])

label_relations = np.array([
    [2, 1],
    [1, 3],
    [3, 2],
    [2, 2]
])

R_size = 4  # Tổng số quy tắc

# Tính các ma trận theo thuật toán FISA
A_matrix = compute_A(X_relations, R_size)
B_matrix = compute_B(A_matrix, label_relations, R_size)
C_matrix = compute_C(B_matrix)
D_matrix = compute_D(C_matrix)
final_label = classify(D_matrix)

# In kết quả
print("Ma trận A:\n", A_matrix)
print("Ma trận B:\n", B_matrix)
print("Ma trận C:\n", C_matrix)
print("Ma trận D:\n", D_matrix)
print("Nhãn đầu ra:", final_label)
