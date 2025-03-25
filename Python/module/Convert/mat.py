from scipy.io import loadmat
import pandas as pd
data = loadmat('D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\FIS\output\Test\RuleList.mat')
print(data.keys())  # Hiển thị các biến trong file

# # Truy cập một biến cụ thể (ví dụ: biến 'X')
X = data['ruleList']
df = pd.DataFrame(X)
df.to_csv('D:\Study\InternAIRC\source_code_Tan\source_code_Tan\Python\data\FIS\output\Test\RuleList_1.csv')
print(X)