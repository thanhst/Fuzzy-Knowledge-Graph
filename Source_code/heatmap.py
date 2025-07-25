import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_image = pd.read_csv("./data/Dataset_diabetic/images_ft.csv")
df_tabular = pd.read_csv("./data/Dataset_diabetic/data_process_tabular.csv")

features_df_image = df_image.drop(columns=['diabetic_retinopathy','image_id'])

features_df_tabular = df_tabular.drop(columns=['diabetic_retinopathy','image_id'],errors='ignore')


corr_image = features_df_image.corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_image, cmap='coolwarm', annot=True, fmt=".2f", square=True, cbar_kws={'shrink': .8})

plt.title("Pearson Correlation Heatmap - Image Features (GLCM)")
plt.tight_layout()
plt.savefig("./data/heatmap/pearson_heatmap_image.png", dpi=300)
plt.close()

corr_tabular = features_df_tabular.corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_tabular, cmap='coolwarm', annot=True, fmt=".2f", square=True, cbar_kws={'shrink': .8})

plt.title("Pearson Correlation Heatmap - Tabular Features")
plt.tight_layout()
plt.savefig("./data/heatmap/pearson_heatmap_tabular.png", dpi=300)
plt.close()
