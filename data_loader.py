"""
data_loader.py
--------------
Handles loading raw and processed data.
"""

import os
import pandas as pd

RAW_PATH       = os.path.join("data", "raw", "OnlineRetail.xlsx")
OFFERS_PATH    = os.path.join("data", "processed", "customer_offers.csv")
SEGMENTS_PATH  = os.path.join("data", "processed", "customer_segments.csv")


def load_raw_data() -> pd.DataFrame:
    """Load the raw Online Retail Excel file."""
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw data not found at '{RAW_PATH}'.\n"
            "Please place OnlineRetail.xlsx in data/raw/"
        )
    df = pd.read_excel(RAW_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def load_offers() -> pd.DataFrame:
    """Load precomputed customer offers CSV."""
    if not os.path.exists(OFFERS_PATH):
        raise FileNotFoundError(
            f"Offers file not found at '{OFFERS_PATH}'.\n"
            "Run the notebook first to generate processed data."
        )
    df = pd.read_csv(OFFERS_PATH)
    # Parse list column stored as string
    if "All_Recommendations" in df.columns:
        import ast
        df["All_Recommendations"] = df["All_Recommendations"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
    return df


def load_segments() -> pd.DataFrame:
    """Load precomputed customer segments CSV."""
    if not os.path.exists(SEGMENTS_PATH):
        raise FileNotFoundError(
            f"Segments file not found at '{SEGMENTS_PATH}'.\n"
            "Run the notebook first to generate processed data."
        )
    return pd.read_csv(SEGMENTS_PATH)


def data_ready() -> bool:
    """Return True if all processed files exist."""
    return os.path.exists(OFFERS_PATH) and os.path.exists(SEGMENTS_PATH)
