"""
charts.py
---------
All Plotly chart builders used across the Streamlit pages.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
SEGMENT_COLORS = {
    "🟢 Active"    : "#2ecc71",
    "⚠️ Inactive"  : "#e74c3c"
}


# ── Dashboard charts ──────────────────────────────────────────────────────────

def segment_pie(offers_df: pd.DataFrame) -> go.Figure:
    counts = offers_df["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    colors = [SEGMENT_COLORS.get(s, "#95a5a6") for s in counts["Segment"]]
    fig = px.pie(
        counts, names="Segment", values="Count",
        color_discrete_sequence=colors,
        hole=0.4,
        title="Customer Segment Distribution",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10))
    return fig


def revenue_by_segment(offers_df: pd.DataFrame) -> go.Figure:
    summary = offers_df.groupby("Segment")["Monetary"].agg(
        Total="sum", Avg="mean", Count="count"
    ).reset_index()
    colors = [SEGMENT_COLORS.get(s, "#95a5a6") for s in summary["Segment"]]
    fig = px.bar(
        summary, x="Segment", y="Total",
        color="Segment", color_discrete_map=SEGMENT_COLORS,
        text_auto=".2s",
        title="Total Revenue by Segment (£)",
        labels={"Total": "Revenue (£)", "Segment": ""},
    )
    fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
    return fig


def discount_bar(offers_df: pd.DataFrame) -> go.Figure:
    grp = offers_df.groupby(["Segment", "Discount_Pct"]).size().reset_index(name="Customers")
    fig = px.bar(
        grp, x="Segment", y="Customers", color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        text="Discount_Pct",
        title="Discount Distribution by Segment",
        labels={"Customers": "# Customers", "Discount_Pct": "Discount %"},
    )
    fig.update_traces(texttemplate="%{text}% off", textposition="inside")
    fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
    return fig


def gmm_confidence_hist(offers_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=offers_df["GMM_Confidence"], nbinsx=30,
        marker_color="#9b59b6", opacity=0.85, name="Confidence"
    ))
    fig.add_vline(x=0.6, line_dash="dash", line_color="red",
                  annotation_text="Borderline (0.6)", annotation_position="top right")
    fig.update_layout(
        title="GMM Assignment Confidence",
        xaxis_title="Max Probability", yaxis_title="Count",
        margin=dict(t=50, b=10),
    )
    return fig


# ── Customer Insights charts ──────────────────────────────────────────────────

def rfm_scatter(rfm: pd.DataFrame) -> go.Figure:
    """Recency vs Monetary scatter coloured by segment."""
    fig = px.scatter(
        rfm, x="Recency", y="Monetary",
        color="Segment", color_discrete_map=SEGMENT_COLORS,
        size="Frequency", size_max=20,
        hover_data=["CustomerID", "Frequency", "UniqueProducts"],
        title="Recency vs Monetary (bubble = Frequency)",
        labels={"Monetary": "Total Spend (£)", "Recency": "Recency (days)"},
        opacity=0.7,
    )
    fig.update_layout(margin=dict(t=50, b=10))
    return fig


def rfm_box(rfm: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.box(
        rfm, x="Segment", y=feature,
        color="Segment", color_discrete_map=SEGMENT_COLORS,
        title=f"{feature} Distribution by Segment",
        points="outliers",
    )
    fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
    return fig


def radar_chart(profile: pd.DataFrame, segment_map: dict) -> go.Figure:
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    cats = list(norm.columns)
    fig  = go.Figure()
    for i, (cluster_id, row) in enumerate(norm.iterrows()):
        vals = list(row.values) + [row.values[0]]
        label = segment_map.get(cluster_id, f"Cluster {cluster_id}")
        color = SEGMENT_COLORS.get(label, PALETTE[i % len(PALETTE)])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=label,
            line_color=color, fillcolor=color,
            opacity=0.25,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Segment Radar Chart (normalized)",
        margin=dict(t=60, b=10),
    )
    return fig


def pca_scatter(pca_2d: np.ndarray, rfm: pd.DataFrame) -> go.Figure:
    tmp = rfm[["KMeans_Cluster", "Segment"]].copy()
    tmp["PC1"] = pca_2d[:, 0]
    tmp["PC2"] = pca_2d[:, 1]
    fig = px.scatter(
        tmp, x="PC1", y="PC2", color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.5, title="Customer Segments — PCA 2D",
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(t=50, b=10))
    return fig


# ── Model Analysis charts ─────────────────────────────────────────────────────

def training_curve(history) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history.history["loss"], name="Train Loss",
        line=dict(color="#3498db", width=2)
    ))
    fig.add_trace(go.Scatter(
        y=history.history["val_loss"], name="Val Loss",
        line=dict(color="#e74c3c", width=2, dash="dash")
    ))
    fig.update_layout(
        title="AutoEncoder Training Loss",
        xaxis_title="Epoch", yaxis_title="MSE",
        margin=dict(t=50, b=10),
    )
    return fig


def elbow_chart(k_range, wcss, sil_scores, db_scores) -> go.Figure:
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["WCSS (Elbow)", "Silhouette ↑", "Davies-Bouldin ↓"])
    fig.add_trace(go.Scatter(x=list(k_range), y=wcss, mode="lines+markers",
                             line=dict(color="#3498db")), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(k_range), y=sil_scores, mode="lines+markers",
                             line=dict(color="#2ecc71")), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(k_range), y=db_scores, mode="lines+markers",
                             line=dict(color="#e74c3c")), row=1, col=3)
    fig.update_layout(title="KMeans — Finding Optimal K",
                      showlegend=False, margin=dict(t=60, b=10))
    return fig


def silhouette_comparison(metrics: dict) -> go.Figure:
    models = list(metrics.keys())
    sils   = [metrics[m]["Silhouette"] for m in models]
    fig    = px.bar(
        x=models, y=sils, color=models,
        color_discrete_sequence=["#3498db", "#e74c3c"],
        text=[f"{s:.4f}" for s in sils],
        title="Silhouette Score Comparison",
        labels={"x": "Model", "y": "Silhouette"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(t=50, b=10), yaxis_range=[0, 0.7])
    return fig


def rec_score_hist(recommendations: dict) -> go.Figure:
    scores = [r[0]["Score"] for r in recommendations.values() if r]
    fig = go.Figure(go.Histogram(
        x=scores, nbinsx=40,
        marker_color="#9b59b6", opacity=0.85,
    ))
    fig.update_layout(
        title="Top Recommendation Score Distribution",
        xaxis_title="AutoEncoder Score", yaxis_title="Count",
        margin=dict(t=50, b=10),
    )
    return fig
