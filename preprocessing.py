"""
preprocessing.py
----------------
All data cleaning and feature engineering steps extracted from the notebook.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


CLUSTER_FEATURES = [
    "Recency", "log_Frequency", "log_Monetary",
    "log_AvgOrderValue", "log_UniqueProducts"
]

NOISE_PATTERN = r"wrong|barcode|adjust|test|sample|lost|damage|manual|found|amazon"


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps from Step 3 of the notebook."""
    df = df_raw.copy()

    # Drop missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Remove cancellations
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove invalid Quantity / UnitPrice
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Fill missing Description
    df["Description"] = df.groupby("StockCode")["Description"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
    )

    # Remove noise descriptions
    df = df[~df["Description"].str.contains(NOISE_PATTERN, case=False, na=False)]
    df = df[~df["Description"].str.match(r"^\d+$", na=False)]
    df = df[df["Description"] != "POSTAGE"]

    # Total price
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Remove upper outliers (IQR)
    for col in ["Quantity", "UnitPrice", "TotalPrice"]:
        df = _remove_upper_outliers(df, col)

    return df.reset_index(drop=True)


def _remove_upper_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - df[col].quantile(0.25)
    return df[df[col] <= Q3 + 1.5 * IQR]


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Build the RFM + extended feature table (Step 4.1)."""
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency        =("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
        Frequency      =("InvoiceNo",    "nunique"),
        Monetary       =("TotalPrice",   "sum"),
        UniqueProducts =("StockCode",    "nunique"),
        TotalItems     =("Quantity",     "sum"),
    ).reset_index()

    rfm["AvgOrderValue"]    = rfm["Monetary"]   / rfm["Frequency"]
    rfm["AvgItemsPerOrder"] = rfm["TotalItems"] / rfm["Frequency"]

    # Log transforms
    for col in ["Monetary", "Frequency", "UniqueProducts",
                "AvgOrderValue", "TotalItems", "AvgItemsPerOrder"]:
        rfm[f"log_{col}"] = np.log1p(rfm[col])

    return rfm


def build_purchase_matrix(df: pd.DataFrame, min_buyers: int = 20):
    """Build Customer × Product interaction matrix (Step 4.6)."""
    product_support = df.groupby("StockCode")["CustomerID"].nunique()
    popular         = product_support[product_support >= min_buyers].index
    df_f            = df[df["StockCode"].isin(popular)]

    interactions = (
        df_f.groupby(["CustomerID", "StockCode"])["Quantity"]
        .sum().reset_index()
    )
    interactions["interaction"] = np.log1p(interactions["Quantity"])

    matrix = interactions.pivot(
        index="CustomerID", columns="StockCode", values="interaction"
    ).fillna(0)

    return matrix


def scale_rfm(rfm: pd.DataFrame, scaler: StandardScaler = None):
    """StandardScale the cluster features. Returns (X_scaled, scaler)."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(rfm[CLUSTER_FEATURES])
    else:
        X_scaled = scaler.transform(rfm[CLUSTER_FEATURES])
    return X_scaled, scaler


def scale_purchase_matrix(X: np.ndarray, scaler: MinMaxScaler = None):
    """MinMaxScale the purchase matrix for the AutoEncoder."""
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled.astype(np.float32), scaler
