import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

st.set_page_config(page_title="SmartCart Customer Segmentation", page_icon="🛒", layout="wide")

SPENDING_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]
DROP_COLS = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"]


# ----------------------------------------------------------------------
# Data pipeline
# ----------------------------------------------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)


@st.cache_data
def preprocess(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # Missing values
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Feature engineering
    df["Age"] = pd.Timestamp.today().year - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
    df["Total_Spending"] = df[SPENDING_COLS].sum(axis=1)
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", "PhD": "Postgraduate",
    })
    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone",
    })

    df_cleaned = df.drop(columns=DROP_COLS + SPENDING_COLS)

    # Outliers
    n_before = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned["Age"] < 90]
    df_cleaned = df_cleaned[df_cleaned["Income"] < 600_000]
    n_removed = n_before - len(df_cleaned)

    # Encoding
    cat_cols = ["Education", "Living_With"]
    ohe = OneHotEncoder()
    enc = ohe.fit_transform(df_cleaned[cat_cols])
    enc_df = pd.DataFrame(
        enc.toarray(),
        columns=ohe.get_feature_names_out(cat_cols),
        index=df_cleaned.index,
    )
    df_encoded = pd.concat([df_cleaned.drop(columns=cat_cols), enc_df], axis=1)

    # Scaling + PCA
    X_scaled = StandardScaler().fit_transform(df_encoded)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    return df_cleaned, df_encoded, X_pca, pca.explained_variance_ratio_, n_removed


@st.cache_data
def k_analysis(X_pca: np.ndarray, max_k: int = 10):
    wcss, sil = [], []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca)
        wcss.append(km.inertia_)
        if k >= 2:
            sil.append(silhouette_score(X_pca, km.labels_))
    knee = KneeLocator(range(1, max_k + 1), wcss, curve="convex", direction="decreasing")
    return wcss, sil, knee.elbow


def fit_clusters(X_pca, algo, k):
    if algo == "KMeans":
        return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_pca)
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_pca)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.title("🛒 SmartCart Customer Segmentation")
st.caption("Unsupervised ML: PCA + KMeans / Agglomerative clustering on customer data")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload CSV (optional)", type="csv")
    algo = st.selectbox("Clustering algorithm", ["Agglomerative", "KMeans"])
    k = st.slider("Number of clusters (K)", 2, 8, 4)
    st.markdown("---")
    st.markdown("Default dataset: `smartcart_customers.csv`")

try:
    df_raw = load_data(uploaded if uploaded else "smartcart_customers.csv")
    df_cleaned, df_encoded, X_pca, var_ratio, n_removed = preprocess(df_raw)
except Exception as e:
    st.error(f"Data load/preprocess error: {e}")
    st.stop()

labels = fit_clusters(X_pca, algo, k)
df_result = df_encoded.copy()
df_result["cluster"] = labels
palette = sns.color_palette("Set1", n_colors=k)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Data", "📊 EDA", "📐 Optimal K", "🧩 Clusters", "👥 Cluster Profile"]
)

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (raw)", df_raw.shape[0])
    c2.metric("Outliers removed", n_removed)
    c3.metric("Rows used", len(df_cleaned))
    st.subheader("Raw data")
    st.dataframe(df_raw.head(20), use_container_width=True)
    st.subheader("Missing values")
    st.dataframe(df_raw.isnull().sum().rename("missing").to_frame().T, use_container_width=True)
    st.subheader("Cleaned data (after feature engineering)")
    st.dataframe(df_cleaned.head(20), use_container_width=True)

with tab2:
    st.subheader("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df_cleaned.corr(numeric_only=True), annot=True, annot_kws={"size": 6},
                cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Feature distributions")
    num_cols = ["Income", "Age", "Total_Spending", "Recency", "Total_Children", "Customer_Tenure_Days"]
    col = st.selectbox("Select feature", num_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_cleaned[col], kde=True, ax=ax)
    st.pyplot(fig)

    if st.checkbox("Show pairplot (slow)"):
        pp = sns.pairplot(df_cleaned[["Income", "Recency", "Response", "Age",
                                      "Total_Spending", "Total_Children"]])
        st.pyplot(pp.figure)

with tab3:
    wcss, sil, elbow = k_analysis(X_pca)
    st.write(f"PCA explained variance (3 components): **{var_ratio.sum():.1%}**")
    st.write(f"Elbow method suggests K = **{elbow}**" if elbow else "Elbow not detected clearly.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker="o")
        ax.set_xlabel("K")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow method")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.plot(range(2, 11), sil, marker="o", color="red")
        ax.set_xlabel("K")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette score")
        st.pyplot(fig)

with tab4:
    st.subheader(f"{algo} clusters (K={k}) in 3D PCA space")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap="Set1", s=10)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    st.pyplot(fig)
    st.metric("Silhouette score (this clustering)", f"{silhouette_score(X_pca, labels):.3f}")

with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cluster sizes")
        fig, ax = plt.subplots()
        sns.countplot(x=df_result["cluster"], hue=df_result["cluster"],
                      palette=palette, legend=False, ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Income vs Spending")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df_result["Total_Spending"], y=df_result["Income"],
                        hue=df_result["cluster"], palette=palette, ax=ax)
        st.pyplot(fig)

    st.subheader("Cluster summary (mean values)")
    st.dataframe(df_result.groupby("cluster").mean().round(2), use_container_width=True)

    st.download_button(
        "⬇️ Download clustered data (CSV)",
        df_result.to_csv(index=False).encode("utf-8"),
        file_name="smartcart_clustered.csv",
        mime="text/csv",
    )
