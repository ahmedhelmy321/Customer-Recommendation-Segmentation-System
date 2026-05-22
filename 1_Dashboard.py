"""
pages/1_Dashboard.py
--------------------
Executive KPI dashboard — segment overview, revenue, offer distribution.
"""

import streamlit as st
import pandas as pd
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import load_offers, load_segments, data_ready
from utils.charts import (
    segment_pie, revenue_by_segment, discount_bar, gmm_confidence_hist
)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Executive Dashboard")
st.markdown("High-level KPIs and segment-level summaries.")

# ── Data check ────────────────────────────────────────────────────────────────
if not data_ready():
    st.error(
        "Processed data not found. Run the notebook first to generate "
        "`data/processed/customer_offers.csv` and `customer_segments.csv`."
    )
    st.stop()

offers_df  = load_offers()
segments_df = load_segments()

# ── Top KPIs ──────────────────────────────────────────────────────────────────
st.markdown("### 📈 Key Metrics")
k1, k2, k3, k4, k5 = st.columns(5)

total_customers   = len(offers_df)
total_revenue     = offers_df["Monetary"].sum()
avg_spend         = offers_df["Monetary"].mean()
borderline        = (offers_df["Segment_Source"] == "GMM-blended (borderline)").sum()
avg_confidence    = offers_df["GMM_Confidence"].mean()

k1.metric("Total Customers",     f"{total_customers:,}")
k2.metric("Total Revenue",       f"£{total_revenue:,.0f}")
k3.metric("Avg Customer Spend",  f"£{avg_spend:,.0f}")
k4.metric("Borderline Customers", f"{borderline:,}")
k5.metric("Avg GMM Confidence",  f"{avg_confidence:.2%}")

st.markdown("---")

# ── Row 1 charts ──────────────────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2)

with r1c1:
    st.plotly_chart(segment_pie(offers_df), use_container_width=True)

with r1c2:
    st.plotly_chart(revenue_by_segment(offers_df), use_container_width=True)

# ── Row 2 charts ──────────────────────────────────────────────────────────────
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.plotly_chart(discount_bar(offers_df), use_container_width=True)

with r2c2:
    st.plotly_chart(gmm_confidence_hist(offers_df), use_container_width=True)

st.markdown("---")

# ── Revenue impact table ───────────────────────────────────────────────────────
st.markdown("### 💰 Estimated Revenue Impact")
st.caption(
    "Assumption-based estimate using industry-benchmark acceptance rates: "
    "Champions 20%, Loyal 25%, At-Risk 15%, Dormant 10%."
)

acceptance = {
    "👑 Champions": 0.20,
    "🟢 Loyal"    : 0.25,
    "⚠️ At-Risk"  : 0.15,
    "💤 Dormant"  : 0.10,
}

rows = []
for seg, grp in offers_df.groupby("Segment"):
    n_cust    = len(grp)
    avg_m     = grp["Monetary"].mean()
    disc      = grp["Discount_Pct"].iloc[0]
    rate      = acceptance.get(seg, 0.15)
    n_accept  = int(n_cust * rate)
    gross     = n_accept * avg_m
    cost      = gross * disc / 100
    net       = gross - cost
    rows.append({
        "Segment"           : seg,
        "Customers"         : n_cust,
        "Discount"          : f"{disc}%",
        "Acceptance Rate"   : f"{int(rate*100)}%",
        "Expected Takers"   : n_accept,
        "Avg Spend"         : f"£{avg_m:,.0f}",
        "Gross Revenue"     : f"£{gross:,.0f}",
        "Discount Cost"     : f"£{cost:,.0f}",
        "Net Revenue"       : f"£{net:,.0f}",
    })

impact_df = pd.DataFrame(rows)
st.dataframe(impact_df, use_container_width=True, hide_index=True)

# ── Raw offers preview ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Offer Table Preview")

display_cols = [
    "CustomerID", "Segment", "Discount_Pct",
    "Recommended_Product", "Rec_Score", "GMM_Confidence",
    "Recency", "Frequency", "Monetary"
]
seg_filter = st.multiselect(
    "Filter by Segment",
    options=offers_df["Segment"].unique().tolist(),
    default=offers_df["Segment"].unique().tolist(),
)
filtered = offers_df[offers_df["Segment"].isin(seg_filter)]
st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True, height=350)
st.caption(f"Showing {len(filtered):,} of {len(offers_df):,} customers.")
