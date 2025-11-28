import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("../plots", exist_ok=True)

df = pd.read_csv("../customers_with_clusters.csv")

def save_plot(fig, name):
    fig.tight_layout()
    fig.savefig(f"../plots/{name}", dpi=150)
    plt.close(fig)

sns.set(style="whitegrid")

# 1. Cluster Count
fig, ax = plt.subplots()
sns.countplot(x=df["Cluster"], ax=ax)
ax.set_title("Cluster Counts")
save_plot(fig, "cluster_count.png")

# 2. Recency Distribution
fig, ax = plt.subplots()
sns.histplot(df["Recency"], bins=30, kde=True, ax=ax)
ax.set_title("Recency Distribution")
save_plot(fig, "recency_distribution.png")

# 3. Frequency Distribution
fig, ax = plt.subplots()
sns.histplot(df["Frequency"], bins=30, kde=True, ax=ax)
ax.set_title("Frequency Distribution")
save_plot(fig, "frequency_distribution.png")

# 4. Monetary Distribution
fig, ax = plt.subplots()
sns.histplot(df["Monetary"], bins=30, kde=True, ax=ax)
ax.set_title("Monetary Distribution")
save_plot(fig, "monetary_distribution.png")

# 5. Recency by Cluster
fig, ax = plt.subplots()
sns.boxplot(x="Cluster", y="Recency", data=df, ax=ax)
ax.set_title("Recency by Cluster")
save_plot(fig, "recency_by_cluster.png")

# 6. Frequency by Cluster
fig, ax = plt.subplots()
sns.boxplot(x="Cluster", y="Frequency", data=df, ax=ax)
ax.set_title("Frequency by Cluster")
save_plot(fig, "frequency_by_cluster.png")

# 7. Monetary vs Frequency
fig, ax = plt.subplots()
sns.scatterplot(x="Frequency", y="Monetary", hue="Cluster", data=df, ax=ax)
ax.set_title("Monetary vs Frequency")
save_plot(fig, "monetary_vs_frequency.png")

# 8. Heatmap
fig, ax = plt.subplots()
sns.heatmap(df[["Recency","Frequency","Monetary"]].corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("RFM Correlation Heatmap")
save_plot(fig, "correlation_heatmap.png")

print("All plots saved in /plots folder!")
