import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/Dataset_diabetic/data_process_fusion.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (using Random Forest)")
plt.barh(range(len(importances)), importances[indices], color="skyblue")
plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("./data/heatmap/Importance.png",dpi=300)
plt.show()
