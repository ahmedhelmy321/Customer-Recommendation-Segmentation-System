"""
pages/2_Customer_Insights.py
-----------------------------
Per-customer deep dive: RFM profile, segment details, offer message, recommendations.
"""

import streamlit as st
import pandas as pd
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import load_offers, load_segments, data_ready
from utils.charts import rfm_scatter, rfm_box, radar_chart, pca_scatter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Insights", page_icon="👤", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("👤 Customer Insights")
st.markdown("Explore individual customer profiles and segment characteristics.")

# ── Data check ────────────────────────────────────────────────────────────────
if not data_ready():
    st.error("Processed data not found. Run the notebook first.")
    st.stop()

offers_df   = load_offers()
segments_df = load_segments()

# Merge for full rfm view
rfm = segments_df.copy()
seg_map_lookup = offers_df[["CustomerID", "Segment"]].drop_duplicates().set_index("CustomerID")["Segment"].to_dict()
rfm["Segment"] = rfm["CustomerID"].map(seg_map_lookup)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Customer Lookup", "📊 Segment Explorer", "🗺️ Segment Map"])

# ── Tab 1: Customer Lookup ────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🔍 Look Up a Customer")

    col_search, col_seg = st.columns([2, 1])
    with col_seg:
        seg_options = ["All"] + sorted(offers_df["Segment"].unique().tolist())
        seg_choice  = st.selectbox("Filter by segment", seg_options)

    with col_search:
        if seg_choice != "All":
            id_options = offers_df[offers_df["Segment"] == seg_choice]["CustomerID"].tolist()
        else:
            id_options = offers_df["CustomerID"].tolist()
        cust_id = st.selectbox("Select Customer ID", id_options)

    if cust_id:
        row = offers_df[offers_df["CustomerID"] == cust_id].iloc[0]

        # ── Profile metrics
        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Segment",     row["Segment"])
        m2.metric("Recency",     f"{int(row['Recency'])} days")
        m3.metric("Frequency",   f"{int(row['Frequency'])} orders")
        m4.metric("Monetary",    f"£{row['Monetary']:,.2f}")
        m5.metric("GMM Conf.",   f"{row['GMM_Confidence']:.2%}")

        # ── Offer details
        st.markdown("---")
        oc1, oc2 = st.columns([1, 2])

        with oc1:
            st.markdown("#### 🎁 Offer Details")
            # st.markdown(f"**Offer Type:** {row['Offer_Type']}")
            st.markdown(f"**Discount:** {row['Discount_Pct']}%")
            st.markdown(f"**Assigned by:** {row['Segment_Source']}")
            st.markdown(f"**Top Product:** {row['Recommended_Product']}")
            if isinstance(row["All_Recommendations"], list) and len(row["All_Recommendations"]) > 1:
                st.markdown("**All Recommendations:**")
                for i, rec in enumerate(row["All_Recommendations"][:5], 1):
                    st.markdown(f"  {i}. {rec}")

        with oc2:
            st.markdown("#### 📩 Personalised Message")
            st.markdown(
                f'<div class="offer-card">{row["Message"]}</div>',
                unsafe_allow_html=True,
            )

# ── Tab 2: Segment Explorer ───────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Segment Deep Dive")

    feat_col, seg_col2 = st.columns([2, 1])
    with seg_col2:
        seg_exp = st.selectbox("Select Segment", offers_df["Segment"].unique().tolist(), key="seg_exp")

    seg_data   = offers_df[offers_df["Segment"] == seg_exp]
    seg_rfm    = rfm[rfm["Segment"] == seg_exp]

    # Stats
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Customers",       f"{len(seg_data):,}")
    sm2.metric("Avg Recency",     f"{seg_rfm['Recency'].mean():.0f} days")
    sm3.metric("Avg Frequency",   f"{seg_rfm['Frequency'].mean():.1f}")
    sm4.metric("Avg Spend",       f"£{seg_rfm['Monetary'].mean():,.0f}")

    st.markdown("---")

    bc1, bc2 = st.columns(2)
    with bc1:
        with feat_col:
            feature = st.selectbox(
                "Box plot feature",
                ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"],
                key="feat_box"
            )
        st.plotly_chart(rfm_box(rfm, feature), use_container_width=True)

    with bc2:
        st.plotly_chart(rfm_scatter(rfm), use_container_width=True)

    # Sample customers in segment
    st.markdown(f"#### Sample customers in **{seg_exp}**")
    sample_cols = ["CustomerID", "Recency", "Frequency", "Monetary",
                   "Discount_Pct", "Recommended_Product", "GMM_Confidence"]
    st.dataframe(
        seg_data[sample_cols].head(20).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )

# ── Tab 3: Segment Map ────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🗺️ Segment Visualization")

    CLUSTER_FEATURES = [
        "Recency", "log_Frequency", "log_Monetary",
        "log_AvgOrderValue", "log_UniqueProducts"
    ]

    has_log = all(c in rfm.columns for c in CLUSTER_FEATURES)
    if has_log:
        X = rfm[CLUSTER_FEATURES].fillna(0).values
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X)
        pca_2d  = PCA(n_components=2).fit_transform(X_sc)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.plotly_chart(pca_scatter(pca_2d, rfm), use_container_width=True)

        with sc2:
            # Radar chart
            PROFILE_COLS = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]
            missing = [c for c in PROFILE_COLS if c not in rfm.columns]
            if not missing:
                profile = rfm.groupby("KMeans_Cluster")[PROFILE_COLS].mean()
                seg_map = rfm.groupby("KMeans_Cluster")["Segment"].first().to_dict()
                st.plotly_chart(radar_chart(profile, seg_map), use_container_width=True)
    else:
        st.info("Log-transformed features not found in segments file. Run notebook with full feature engineering.")
