from module.Convert.var_lang import change_var_lang̣
import numpy as np
rules = [1,2,3,1,2,3,5]
lang_matrix = [["Low","Medium","High"],["Low","Medium","High"],["Low","Medium","High"],["Low","Medium","High"],["Low","Medium","High"],["Low","Medium","High"]]
rules = np.array(rules)
rules = rules.reshape(1,-1)
data = change_var_lang̣(lang_matrix,rules)
print(data)
