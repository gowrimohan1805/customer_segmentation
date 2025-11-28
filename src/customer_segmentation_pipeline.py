import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from joblib import dump

# ---------------------------------------------
# Load Dataset
# ---------------------------------------------
df = pd.read_csv("../cleaned_online_retail.csv")

# ---------------------------------------------
# Create RFM Features
# ---------------------------------------------
rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (df["InvoiceDate"].max() - x.max()).days,
    "Invoice": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

# Save RFM
rfm.to_csv("../rfm_features.csv", index=False)

# ---------------------------------------------
# Scaling
# ---------------------------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

# ---------------------------------------------
# PCA
# ---------------------------------------------
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

# ---------------------------------------------
# KMeans Clustering
# ---------------------------------------------
model = MiniBatchKMeans(n_clusters=4, random_state=42)
clusters = model.fit_predict(rfm_pca)

rfm["Cluster"] = clusters
rfm.to_csv("../customers_with_clusters.csv", index=False)

# ---------------------------------------------
# Save Preprocessor + Model
# ---------------------------------------------
dump({
    "scaler": scaler,
    "pca": pca,
    "kmeans": model
}, "../preprocessor_and_model.joblib")

print("Pipeline completed successfully!")
