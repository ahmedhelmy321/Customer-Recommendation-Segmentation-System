"""
segmentation.py
---------------
KMeans + GMM segmentation logic (Steps 5.x from notebook).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.optimize import linear_sum_assignment


RANDOM_STATE = 42
BEST_K       = 4

PROFILE_COLS = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]


# ── KMeans ────────────────────────────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11)):
    """Return WCSS, silhouette, and DB scores for each K."""
    wcss, sil, db = [], [], []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=15, max_iter=500)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, labels))
        db.append(davies_bouldin_score(X_scaled, labels))
    return list(k_range), wcss, sil, db


def train_kmeans(X_scaled: np.ndarray, k: int = BEST_K) -> KMeans:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    km.fit(X_scaled)
    return km


def auto_label_clusters(profile: pd.DataFrame, global_median: pd.Series) -> dict:
    """Map cluster id → segment label based on RFM profile logic."""
    def _label(row):
        r, f, m = row["Recency"], row["Frequency"], row["Monetary"]
        if r < global_median["Recency"] and f > global_median["Frequency"] and m > global_median["Monetary"]:
            return "👑 Champions"
        elif r < global_median["Recency"] and f >= global_median["Frequency"]:
            return "🟢 Loyal"
        elif r > global_median["Recency"] * 1.5:
            return "💤 Dormant"
        else:
            return "⚠️ At-Risk"
    return profile.apply(_label, axis=1).to_dict()


# ── GMM ───────────────────────────────────────────────────────────────────────

def train_gmm(X_scaled: np.ndarray, kmeans: KMeans, k: int = BEST_K) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components    = k,
        covariance_type = "full",
        random_state    = RANDOM_STATE,
        n_init          = 5,
        means_init      = kmeans.cluster_centers_,
    )
    gmm.fit(X_scaled)
    return gmm


def align_gmm_to_kmeans(rfm: pd.DataFrame) -> pd.DataFrame:
    """Re-align GMM cluster IDs so they match KMeans IDs by majority vote."""
    conf      = pd.crosstab(rfm["KMeans_Cluster"], rfm["GMM_Cluster"])
    row_ind, col_ind = linear_sum_assignment(-conf.values)
    gmm_to_km = dict(zip(col_ind, row_ind))
    rfm["GMM_Cluster_aligned"] = rfm["GMM_Cluster"].map(gmm_to_km)
    return rfm


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_clustering(X_scaled: np.ndarray, rfm: pd.DataFrame) -> dict:
    km_sil = silhouette_score(X_scaled,  rfm["KMeans_Cluster"])
    km_db  = davies_bouldin_score(X_scaled, rfm["KMeans_Cluster"])
    km_ch  = calinski_harabasz_score(X_scaled, rfm["KMeans_Cluster"])
    gm_sil = silhouette_score(X_scaled,  rfm["GMM_Cluster"])
    gm_db  = davies_bouldin_score(X_scaled, rfm["GMM_Cluster"])
    gm_ch  = calinski_harabasz_score(X_scaled, rfm["GMM_Cluster"])
    return {
        "KMeans": {"Silhouette": km_sil, "Davies-Bouldin": km_db, "Calinski-Harabasz": km_ch},
        "GMM":    {"Silhouette": gm_sil, "Davies-Bouldin": gm_db, "Calinski-Harabasz": gm_ch},
    }
