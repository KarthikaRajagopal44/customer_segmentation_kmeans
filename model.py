import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

df= pd.read_csv("wholesale_customers_data.csv")
print(df.head())

#scale the features
X= df.drop(["Channel", "Region"] ,axis=1)
scaler= StandardScaler()
X_scaled= scaler.fit_transform(X)

print("\nCOLUMNS AFTER DROPPING:")
print(X.columns)

# Apply KMeans (final model)
km = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X_scaled)

# PCA visualization
pca = PCA(n_components=2)
X_PCA = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_PCA[:,0], y=X_PCA[:,1], hue=df["cluster"], palette="Set1")
plt.title("PCA")
plt.show()

# Print metrics
print("Silhouette Score:", silhouette_score(X_scaled, df['cluster']))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, df['cluster']))
print("Calinski-Harabasz Index:", calinski_harabasz_score(X_scaled, df['cluster']))

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(km, "models/kmeans.pkl")
joblib.dump(scaler, "models/scaler.pkl")
