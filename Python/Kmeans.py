import numpy as np
from module.Kmeans.FIS import FIS
import pandas as pd
from module.Kmeans.Conflict_handling import conflict_handling
df = pd.read_csv("./data/FIS/train_data.csv")
df1=df.drop(df.columns[0],axis=1)
k_values = [3,3,3,3,3,3]
lang_f1 = ["Low","Medium","High"]
lang_f2 = ["Low","Medium","High"]
lang_f3 = ["Low","Medium","High"]
lang_f4 = ["Low","Medium","High"]
lang_f5 = ["Low","Medium","High"]
lang_f6 = ["Low","Medium","High"]
vals_lang = [lang_f1,lang_f2,lang_f3,lang_f4,lang_f5,lang_f6]

f_df1 = FIS(df1,k_values,vals_lang)
f_df1.to_csv("./data/Kmeans/Kmeans_i.csv",index=False)
f_df1 = conflict_handling(f_df1)
f_df1.to_csv("./data/Kmeans/Result_kmeans.csv",index=False)