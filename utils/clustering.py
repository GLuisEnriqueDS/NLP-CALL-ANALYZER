import numpy as np
from sklearn.cluster import KMeans
from utils.config import embed_model

def generate_embeddings(texts):
    return embed_model.encode(
        texts, 
        batch_size=64, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )

def cluster_embeddings(df, k):
    print(f"Aplicando KMeans (k={k})...")
    X = np.vstack(df["embedding"].values)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X)
    return df, kmeans

def top_n_representatives(df_cluster, kmeans, cluster_id, n=1):
    X = np.vstack(df_cluster["embedding"].values)
    centroid = kmeans.cluster_centers_[cluster_id]
    distances = np.linalg.norm(X - centroid, axis=1)

    df_cluster = df_cluster.copy()
    df_cluster["dist"] = distances
    df_cluster = df_cluster.sort_values("dist")

    top_rows = df_cluster.head(n)
    return top_rows.index.tolist(), top_rows["transcripcion"].tolist()
