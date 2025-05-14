import pandas as pd

df = pd.read_csv('../data/table_features/table_ft.csv')

df['dr_class'] = df['dr_class'].replace(2, 1)

df.to_csv('../data/table_features/table_ft.csv', index=False)
