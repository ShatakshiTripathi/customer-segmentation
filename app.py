# app.py
import streamlit as st
import pandas as pd
import numpy as np

# ✅ Safe import for matplotlib
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    st.error("Matplotlib is not installed. Please check requirements.txt and redeploy the app.")
    st.stop()

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------
# Title and Introduction
# --------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Customer Segmentation using K-Means Clustering")
st.markdown("""
Upload your customer dataset (CSV), select features, and this app will:
1. Perform clustering using K-Means.
2. Show cluster visualizations and profiles.
3. Allow you to predict cluster for a **new customer**.
""")

# --------------------------
# File Upload
# --------------------------
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    st.info("Using sample dataset (Mall Customers)...")
    df = pd.DataFrame({
        "CustomerID": range(1, 13),
        "Gender": ["Male", "Male", "Female", "Female", "Female", "Female", "Female", "Male", "Male", "Female", "Male", "Female"],
        "Age": [19, 21, 20, 23, 31, 22, 35, 23, 64, 30, 67, 35],
        "AnnualIncome_k": [15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20],
        "SpendingScore": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 14, 99]
    })

# --------------------------
# Data Preview
# --------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------
# Feature Selection
# --------------------------
st.sidebar.header("Feature Selection")
all_features = df.select_dtypes(include=[np.number]).columns.tolist()
selected_features = st.sidebar.multiselect("Select features for clustering", all_features, default=all_features)

if len(selected_features) < 2:
    st.warning("⚠️ Please select at least 2 features for clustering.")
    st.stop()

X = df[selected_features]

# --------------------------
# Data Standardization
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# Choosing number of clusters (Elbow method)
# --------------------------
st.subheader("📈 Elbow Method to choose k")
inertia = []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

fig, ax = plt.subplots()
ax.plot(K, inertia, 'bo-')
ax.set_xlabel('Number of clusters k')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method')
st.pyplot(fig)

# --------------------------
# User selects number of clusters
# --------------------------
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

# --------------------------
# KMeans Clustering
# --------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.subheader("🧩 Cluster Results")
st.write(df.head())

# --------------------------
# Cluster Distribution
# --------------------------
st.subheader("📊 Cluster Distribution")
st.bar_chart(df['Cluster'].value_counts())

# --------------------------
# Cluster Profiles
# --------------------------
st.subheader("📋 Cluster Profiles")
cluster_profiles = df.groupby('Cluster')[selected_features].mean()
st.dataframe(cluster_profiles)

# --------------------------
# PCA for 2D Visualization
# --------------------------
st.subheader("🎨 PCA Visualization of Clusters")
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df['Cluster']

fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100, ax=ax)
st.pyplot(fig)

# --------------------------
# Predict Cluster for New Customer
# --------------------------
st.sidebar.header("🔮 Predict New Customer")
input_data = []
for feature in selected_features:
    value = st.sidebar.number_input(
        f"Enter {feature}",
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean())
    )
    input_data.append(value)

if st.sidebar.button("Predict Cluster"):
    input_scaled = scaler.transform([input_data])
    cluster = kmeans.predict(input_scaled)[0]
    st.sidebar.success(f"Predicted Cluster: {cluster}")
