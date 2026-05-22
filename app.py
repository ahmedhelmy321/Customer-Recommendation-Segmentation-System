"""
app.py
------
Main entry point for the Customer Recommendation System Streamlit app.
Run with:  streamlit run app.py
"""

import streamlit as st
import os
import sys

# ── Ensure project root is in path so 'utils' is always found ────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 RetailAI")
    st.markdown("**Customer Recommendation System**")
    st.markdown("---")
    st.markdown(
        """
        **Navigation**
        - 📊 Dashboard — KPIs & overview
        - 👤 Customer Insights — segment deep-dives
        - 🧠 Model Analysis — AutoEncoder & clustering metrics
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Built with Streamlit · TensorFlow · scikit-learn")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.title("🛒 Customer Recommendation System")
st.markdown(
    """
    **A full ML pipeline** that segments ~4,300 UK retail customers into behavioural
    groups using **KMeans + GMM**, generates personalised product recommendations
    via an **AutoEncoder**, and delivers tailored discount offers per segment.
    """
)

st.markdown("---")

# ── Pipeline overview ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏗️ Data & Features")
    st.markdown(
        """
        - 541K raw transactions → 270K clean rows
        - RFM + 3 extended features per customer
        - Customer × Product interaction matrix
        - 98.3% sparse — handled by AutoEncoder
        """
    )

with col2:
    st.markdown("### 🤖 Models")
    st.markdown(
        """
        - **KMeans** (K=2): hard segment assignment
        - **GMM**: soft probabilistic membership
        - **AutoEncoder** (256→128→64→128→256): collaborative filtering
        - EarlyStopping + ReduceLROnPlateau
        """
    )

with col3:
    st.markdown("### 🎁 Output")
    st.markdown(
        """
        - 2 customer segments: Active / Inactive
        - Top-5 novel product recommendations per customer
        - Personalised offer message + discount per segment
        - Borderline customers receive GMM-blended offers
        """
    )

st.markdown("---")
# ── Data status ───────────────────────────────────────────────────────────────
from utils.data_loader import data_ready  # noqa: E402  (imported after sys.path fix)

if data_ready():
    st.success(
        "Processed data found — all pages are fully interactive.",
        icon="✅",
    )
else:
    st.warning(
        "Processed data not found.\n\n"
        "Please run the notebook **notebooks/Recommendation_System_Pipeline.ipynb** first "
        "to generate `data/processed/customer_offers.csv` and `customer_segments.csv`.",
        icon="⚠️",
    )
