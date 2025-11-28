# 🛒 Customer Segmentation using RFM + K-Means  
Machine Learning · Python · Data Science · E-Commerce Analytics  

This project performs **Customer Segmentation** using the classic **Online Retail dataset**.  
It applies **RFM (Recency, Frequency, Monetary)** feature engineering and **K-Means clustering**  
to identify customer groups such as:

- 🥇 **VIP / High-Value Customers**  
- 🔁 **Frequent Buyers**  
- 💸 **Low-Monetary Customers**  
- ❌ **At-Risk / Churn Customers**

All analysis, processing, modeling, and visualizations are included.

---

## 📁 Project Structure


```
customer_segmentation_complete/
│
├── online_retails.csv                 # Raw dataset (input)
├── cleaned_online_retail.csv          # Cleaned dataset
├── rfm_features.csv                   # RFM metrics per customer
├── customers_with_clusters.csv        # Final customer clusters
├── preprocessor_and_model.joblib      # Scaler + PCA + KMeans pipeline
│
├── customer_segmentation.ipynb        # End-to-end workflow notebook
├── customer_segmentation_plots.ipynb  # Notebook generating 10 plots
│
├── plots/                             # Saved visualizations
│     ├── cluster_count.png
│     ├── recency_distribution.png
│     ├── frequency_distribution.png
│     ├── monetary_distribution.png
│     ├── recency_by_cluster.png
│     ├── frequency_by_cluster.png
│     ├── monetary_vs_frequency.png
│     ├── correlation_heatmap.png
│     ├── monetary_by_cluster.png
│     └── rfm_pairplot.png
│
├── requirements.txt                   # Python dependencies
└── README.md                          # Complete project documentation
```


---

## 🔍 **Project Overview**

### **1️⃣ Data Cleaning**
- Removed invalid rows  
- Filtered negative quantities (returns)
- Parsed `InvoiceDate`
- Calculated `TotalPrice = Quantity * Price`

### **2️⃣ RFM Feature Engineering**
RFM metrics were calculated for each customer:

| Metric     | Meaning |
|------------|---------|
| **Recency** | Days since last purchase |
| **Frequency** | Number of invoices |
| **Monetary** | Total spend |

Added `log` transformation to fix skew in Monetary values.

---

## 🤖 **Clustering (K-Means)**

- StandardScaler used to scale RFM features  
- MiniBatchKMeans used for efficiency  
- Optimal k selected using **silhouette score**  
- PCA applied for 2D visualization  
- Final dataset saved as `customers_with_clusters.csv`

---

## 📊 **Visualizations (10 Plots)**  
All saved under **plots/** folder.

Includes:

- Cluster counts  
- Recency/Frequency/Monetary distributions  
- Boxplots by cluster  
- RFM heatmap  
- Scatter: Monetary vs Frequency  
- Pairplot colored by cluster  
- PCA-based cluster scatter  

---

## ▶️ **How to Run the Project**

### **1. Install dependencies**
```bash
pip install -r requirements.txt
