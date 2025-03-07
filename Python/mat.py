from scipy.io import loadmat

data = loadmat('./data/BaseData/KGF-Cucdoan/output/RuleList.mat')  # Đọc file .mat
print(data.keys())  # Hiển thị các biến trong file

# Truy cập một biến cụ thể (ví dụ: biến 'X')
X = data['ruleList']
print(X)