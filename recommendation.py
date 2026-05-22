"""
recommendation.py
-----------------
AutoEncoder model definition, training, recommendation generation,
and personalized offer generation (Steps 6–7 of notebook).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


tf.random.set_seed(42)
np.random.seed(42)

TOP_N = 5

OFFER_TEMPLATES = {
    "👑 Champions": {
        "discount_pct"    : 5,
        "offer_type"      : "VIP Early Access",
        "message_template": (
            "Dear Valued Customer, as one of our top Champions, "
            "we're thrilled to offer you exclusive early access to our new collection "
            "plus 5%% off your next order. We think you'll love: \"{product}\"."
        ),
    },
    "🟢 Loyal": {
        "discount_pct"    : 10,
        "offer_type"      : "Loyalty Reward",
        "message_template": (
            "Thank you for being a loyal customer! "
            "Enjoy 10%% off your next purchase as our way of saying thank you. "
            "Based on your taste, we recommend: \"{product}\"."
        ),
    },
    "⚠️ At-Risk": {
        "discount_pct"    : 15,
        "offer_type"      : "Win-Back Campaign",
        "message_template": (
            "We miss you! It's been a while since your last visit. "
            "Come back today with 15%% off — this offer expires in 7 days. "
            "Don't miss our top pick for you: \"{product}\"."
        ),
    },
    "💤 Dormant": {
        "discount_pct"    : 20,
        "offer_type"      : "Re-Engagement Offer",
        "message_template": (
            "Surprise! We have a special 20%% welcome-back discount just for you. "
            "We've missed having you as a customer. "
            "Check out something we think you'll love: \"{product}\"."
        ),
    },
}


# ── Model ─────────────────────────────────────────────────────────────────────

def build_autoencoder(input_dim: int, encoding_dim: int = 64):
    """Build and compile the AutoEncoder + Encoder models."""
    inputs = Input(shape=(input_dim,), name="purchase_vector")

    x = Dense(256, activation="relu", name="enc_1")(inputs)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.3, name="drop_1")(x)

    x = Dense(128, activation="relu", name="enc_2")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(0.2, name="drop_2")(x)

    encoded = Dense(encoding_dim, activation="relu",
                    activity_regularizer=l2(1e-5),
                    name="bottleneck")(x)

    x = Dense(128, activation="relu", name="dec_1")(encoded)
    x = BatchNormalization(name="bn_3")(x)

    x = Dense(256, activation="relu", name="dec_2")(x)
    x = BatchNormalization(name="bn_4")(x)

    decoded = Dense(input_dim, activation="sigmoid", name="output")(x)

    autoencoder = Model(inputs, decoded,  name="AutoEncoder")
    encoder     = Model(inputs, encoded, name="Encoder")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )
    return autoencoder, encoder


def train_autoencoder(autoencoder, X_ae: np.ndarray, epochs: int = 100,
                      batch_size: int = 64, validation_split: float = 0.1):
    """Train the autoencoder with early stopping."""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    history = autoencoder.fit(
        X_ae, X_ae,
        epochs           = epochs,
        batch_size       = batch_size,
        validation_split = validation_split,
        callbacks        = callbacks,
        verbose          = 0,
    )
    return history


# ── Recommendations ───────────────────────────────────────────────────────────

def generate_recommendations(autoencoder, X_ae: np.ndarray,
                              cust_order, product_index: list,
                              product_name_map: dict, top_n: int = TOP_N) -> dict:
    """Return dict: customer_id → list of top_n recommendation dicts."""
    X_rec   = autoencoder.predict(X_ae, verbose=0)
    already = X_ae > 0
    X_masked = X_rec.copy()
    X_masked[already] = -1

    recommendations = {}
    for i, cust_id in enumerate(cust_order):
        scores   = X_masked[i]
        top_idx  = np.argsort(scores)[::-1][:top_n]
        recs     = [
            {
                "StockCode"  : product_index[j],
                "Description": product_name_map.get(product_index[j], "Unknown"),
                "Score"      : float(scores[j]),
            }
            for j in top_idx if scores[j] > 0
        ]
        recommendations[cust_id] = recs
    return recommendations


# ── Offer Generation ──────────────────────────────────────────────────────────

def generate_offer(customer_id, rfm_row, recs: list,
                   gmm_probs_row: np.ndarray, segment_map: dict) -> dict:
    """Generate a personalized offer dict for one customer."""
    gmm_confidence = rfm_row["GMM_MaxProb"]
    kmeans_segment = rfm_row["Segment"]

    if gmm_confidence < 0.6:
        best_cluster = int(np.argmax(gmm_probs_row))
        segment = segment_map[best_cluster]
        note    = "GMM-blended (borderline)"
    else:
        segment = kmeans_segment
        note    = "KMeans (confident)"

    top_rec      = recs[0] if recs else None
    product_name = top_rec["Description"] if top_rec else "Our best sellers"
    product_code = top_rec["StockCode"]   if top_rec else "N/A"
    rec_score    = top_rec["Score"]       if top_rec else 0.0

    template = OFFER_TEMPLATES.get(segment, OFFER_TEMPLATES["🟢 Loyal"])
    message  = template["message_template"] % {"product": product_name}

    return {
        "CustomerID"         : customer_id,
        "Segment"            : segment,
        "Segment_Source"     : note,
        "GMM_Confidence"     : round(gmm_confidence, 4),
        "Discount_Pct"       : template["discount_pct"],
        "Offer_Type"         : template["offer_type"],
        "Recommended_Code"   : product_code,
        "Recommended_Product": product_name,
        "Rec_Score"          : round(rec_score, 5),
        "All_Recommendations": [r["Description"] for r in recs],
        "Message"            : message,
        "Recency"            : rfm_row["Recency"],
        "Frequency"          : rfm_row["Frequency"],
        "Monetary"           : round(rfm_row["Monetary"], 2),
    }


def generate_all_offers(rfm: pd.DataFrame, recommendations: dict,
                        gmm_probs: np.ndarray, segment_map: dict) -> pd.DataFrame:
    """Generate offers for all customers."""
    rfm_idx = rfm.set_index("CustomerID")
    rows = []
    for i, cust_id in enumerate(rfm["CustomerID"].values):
        row   = rfm_idx.loc[cust_id]
        recs  = recommendations.get(cust_id, [])
        offer = generate_offer(cust_id, row, recs, gmm_probs[i], segment_map)
        rows.append(offer)
    return pd.DataFrame(rows)


# ── Metrics ───────────────────────────────────────────────────────────────────

def recommendation_metrics(recommendations: dict, product_index: list,
                            X_ae: np.ndarray, X_rec: np.ndarray) -> dict:
    user_cov = sum(1 for v in recommendations.values() if v) / len(recommendations)
    all_codes = {r["StockCode"] for recs in recommendations.values() for r in recs}
    cat_cov   = len(all_codes) / len(product_index)
    recon_mse = float(np.mean((X_ae - X_rec) ** 2))
    avg_recs  = float(np.mean([len(v) for v in recommendations.values()]))
    return {
        "User Coverage"    : user_cov,
        "Catalog Coverage" : cat_cov,
        "Novelty"          : 1.0,
        "Avg Recs/Customer": avg_recs,
        "Reconstruction MSE": recon_mse,
    }
