"""
pages/3_Model_Analysis.py
--------------------------
Model metrics: clustering evaluation, AutoEncoder analysis, recommendation quality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import load_offers, load_segments, data_ready
from utils.charts import silhouette_comparison, rec_score_hist, elbow_chart

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Analysis", page_icon="🧠", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧠 Model Analysis")
st.markdown("AutoEncoder quality, clustering metrics, and recommendation system evaluation.")

# ── Data check ────────────────────────────────────────────────────────────────
if not data_ready():
    st.error("Processed data not found. Run the notebook first.")
    st.stop()

offers_df   = load_offers()
segments_df = load_segments()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📐 Clustering", "🔮 AutoEncoder", "🎯 Recommendations"])

# ── Tab 1: Clustering ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📐 Clustering Evaluation")

    st.markdown(
        """
        We evaluate clustering quality using three complementary metrics:

        | Metric | Direction | What it measures |
        |---|---|---|
        | **Silhouette** | ↑ higher better | How similar an object is to its own cluster vs. other clusters |
        | **Davies-Bouldin** | ↓ lower better | Average ratio of within-cluster to between-cluster distances |
        | **Calinski-Harabasz** | ↑ higher better | Ratio of between-cluster to within-cluster dispersion |
        """
    )

    # Computed from segments file if cluster columns exist
    CLUSTER_FEATURES = [
        "Recency", "log_Frequency", "log_Monetary",
        "log_AvgOrderValue", "log_UniqueProducts"
    ]
    has_features = all(c in segments_df.columns for c in CLUSTER_FEATURES)

    if has_features and "KMeans_Cluster" in segments_df.columns and "GMM_Cluster" in segments_df.columns:
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        X     = segments_df[CLUSTER_FEATURES].fillna(0).values
        sc    = StandardScaler()
        X_sc  = sc.fit_transform(X)

        km_labels  = segments_df["KMeans_Cluster"].values
        gmm_labels = segments_df["GMM_Cluster"].values

        km_sil = silhouette_score(X_sc, km_labels)
        km_db  = davies_bouldin_score(X_sc, km_labels)
        km_ch  = calinski_harabasz_score(X_sc, km_labels)
        gm_sil = silhouette_score(X_sc, gmm_labels)
        gm_db  = davies_bouldin_score(X_sc, gmm_labels)
        gm_ch  = calinski_harabasz_score(X_sc, gmm_labels)

        metrics = {
            "KMeans": {"Silhouette": km_sil, "Davies-Bouldin": km_db, "Calinski-Harabasz": km_ch},
            "GMM":    {"Silhouette": gm_sil, "Davies-Bouldin": gm_db, "Calinski-Harabasz": gm_ch},
        }

        # Metric cards
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("KMeans Silhouette",  f"{km_sil:.4f}", f"vs GMM {gm_sil:.4f}")
        mc2.metric("KMeans Davies-Bouldin", f"{km_db:.4f}", f"vs GMM {gm_db:.4f}", delta_color="inverse")
        mc3.metric("KMeans CH Score",    f"{km_ch:.1f}", f"vs GMM {gm_ch:.1f}")

        st.markdown("---")
        st.plotly_chart(silhouette_comparison(metrics), use_container_width=True)

        # Metrics table
        rows = []
        for model, vals in metrics.items():
            rows.append({"Model": model, **{k: round(v, 4) for k, v in vals.items()}})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Agreement
        if "GMM_Cluster_aligned" in segments_df.columns or "GMM_Cluster" in segments_df.columns:
            agreement = (segments_df["KMeans_Cluster"] == segments_df["GMM_Cluster"]).mean()
            st.info(f"**KMeans ↔ GMM label agreement:** {agreement:.2%}  "
                    f"({int((1-agreement)*len(segments_df)):,} borderline customers)")
    else:
        st.warning(
            "Cluster columns or log-transformed features not found in segments CSV. "
            "Make sure you export the full `rfm` DataFrame from the notebook."
        )

    # Elbow chart placeholder
    st.markdown("---")
    st.markdown("### K Selection — Elbow Analysis")
    st.info(
        "The elbow + silhouette analysis below is from the notebook. "
        "Re-run the notebook to see updated K-selection plots. "
        "We selected **K=4** for interpretable business segments: "
        "Champions / Loyal / At-Risk / Dormant."
    )

# ── Tab 2: AutoEncoder ────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🔮 AutoEncoder Architecture")

    st.markdown(
        """
        ```
        Input: customer purchase vector  (n_products ≈ 1,200 dims)
             ↓ Dense(256) + BN + Dropout(0.3)
             ↓ Dense(128) + BN + Dropout(0.2)
             ↓ Dense(64)  [Bottleneck — L2 regularized]
             ↓ Dense(128) + BN
             ↓ Dense(256) + BN
        Output: reconstructed purchase vector (sigmoid)
        ```

        **Key design choices:**
        - `log1p(quantity)` input dampens bulk-purchase dominance
        - MinMaxScaler to [0,1] → sigmoid output aligned
        - Masking already-purchased products ensures **novelty = 100%**
        - EarlyStopping (patience=10) + ReduceLROnPlateau (patience=5)
        """
    )

    ae1, ae2, ae3 = st.columns(3)

    ae_model_path = os.path.join("models", "autoencoder.keras")
    model_exists  = os.path.exists(ae_model_path)

    ae1.metric("Encoding Dim",    "64")
    ae2.metric("Activation",      "ReLU / Sigmoid")
    ae3.metric("Loss Function",   "MSE")

    st.markdown("---")
    st.markdown("### Training Metrics")

    if model_exists:
        st.success("✅ Saved AutoEncoder model found at `models/autoencoder.keras`")
    else:
        st.warning(
            "No saved AutoEncoder found. After running the notebook, save the model with:\n"
            "```python\nautoencoder.save('models/autoencoder.keras')\n```"
        )

    st.markdown(
        """
        #### Evaluation metrics (from notebook run)

        | Metric | Description |
        |---|---|
        | **Reconstruction MSE** | How well the AE reconstructs known purchase patterns |
        | **Val Loss** | MSE on 10% held-out customers during training |
        | **User Coverage** | % of customers with ≥1 recommendation |
        | **Catalog Coverage** | % of products recommended at least once |
        | **Novelty** | 100% — masked already-purchased items by design |
        """
    )

# ── Tab 3: Recommendations ────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🎯 Recommendation System Metrics")

    rc1, rc2, rc3, rc4 = st.columns(4)

    rec_col = "All_Recommendations"
    has_recs = rec_col in offers_df.columns

    if has_recs:
        covered       = (offers_df[rec_col].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)).sum()
        user_cov      = covered / len(offers_df)
        avg_recs      = offers_df[rec_col].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()

        rc1.metric("User Coverage",        f"{user_cov:.1%}")
        rc2.metric("Avg Recs / Customer",  f"{avg_recs:.2f}")
        rc3.metric("Novelty",              "100%")
        rc4.metric("Total Offers",         f"{len(offers_df):,}")

        st.markdown("---")

        # Rec score histogram
        if "Rec_Score" in offers_df.columns:
            import plotly.express as px
            fig = px.histogram(
                offers_df[offers_df["Rec_Score"] > 0],
                x="Rec_Score", nbins=40,
                color_discrete_sequence=["#9b59b6"],
                title="Top Recommendation Score Distribution",
                labels={"Rec_Score": "AutoEncoder Score"},
                opacity=0.85,
            )
            fig.update_layout(margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Top recommended products
        st.markdown("### 🏆 Most Recommended Products (across all customers)")
        all_recs = []
        for recs in offers_df[rec_col]:
            if isinstance(recs, list):
                all_recs.extend(recs)
        if all_recs:
            top_prods = pd.Series(all_recs).value_counts().head(15).reset_index()
            top_prods.columns = ["Product", "Times Recommended"]
            import plotly.express as px
            fig2 = px.bar(
                top_prods, x="Times Recommended", y="Product",
                orientation="h",
                color="Times Recommended",
                color_continuous_scale="Blues",
                title="Top 15 Most Recommended Products",
            )
            fig2.update_layout(yaxis=dict(autorange="reversed"),
                               margin=dict(t=50, b=10), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("All_Recommendations column not found in offers CSV.")

    # Offer type breakdown
    st.markdown("---")
    st.markdown("### 📊 Offer Type Breakdown")

    if "Offer_Type" in offers_df.columns:
        ot_counts = offers_df["Offer_Type"].value_counts().reset_index()
        ot_counts.columns = ["Offer Type", "Count"]
        import plotly.express as px
        fig3 = px.bar(
            ot_counts, x="Offer Type", y="Count",
            color="Offer Type",
            text_auto=True,
            title="Customers per Offer Type",
        )
        fig3.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig3, use_container_width=True)
