# 🛒 SmartCart Customer Segmentation

Streamlit web app that segments supermarket customers using unsupervised machine learning.

**🔗 Live demo:** https://YOUR-APP-NAME.streamlit.app  <!-- yahan apna link daalo -->

## Features
- Data cleaning, feature engineering (Age, Tenure, Total Spending, Total Children)
- Outlier removal, One-Hot Encoding, Standard Scaling
- PCA (3 components) for dimensionality reduction
- Optimal K using Elbow method + Silhouette score
- KMeans and Agglomerative (Ward) clustering with 3D visualization
- Cluster profiling + downloadable clustered dataset

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Kneed

## Run locally
```bash
git clone https://github.com/YOUR-USERNAME/smartcart-customer-segmentation.git
cd smartcart-customer-segmentation
pip install -r requirements.txt
streamlit run app.py
```
