#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py — KMeans model training, loading, inference, and drift detection logic.

This module implements:
    • User behavior profiling (mean/std login hour)
    • Feature extraction orchestration
    • KMeans training with auto-K selection
    • Outlier scoring (multi-layer composite score)
    • Drift detection using Chi-square distance

It is used by the main pipeline:
    etc/pipeline.py
"""



import os
import json
from datetime import datetime
from typing import List, Dict

import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.stats import chisquare

from core.features import extract_features
from core.utils import parse_windows_time


# ============================================================================
#  MODEL EXISTENCE CHECK
# ============================================================================

def model_exists(path: str) -> bool:
    """
    Check whether a model already exists on disk.

    A valid model consists of:
        <path>.pkl          → trained KMeans model
        <path>_scaler.pkl   → fitted MinMaxScaler
        <path>.json         → metadata

    Only checks for the .pkl (KMeans) file.

    Parameters
    ----------
    path : str
        Base filepath (without extension).

    Returns
    -------
    bool
        True if model exists, else False.
    """
    return os.path.exists(path + ".pkl")


# ============================================================================
#  BUILD USER PROFILE (mean/std login hour)
# ============================================================================

def build_user_profile(events: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Build per-user behavioral statistics:
        user → { mean_hour, std_hour }

    The login hour mean and std help capture long-term temporal
    patterns of user authentication behavior.

    Parameters
    ----------
    events : list of dict
        List of authentication events.

    Returns
    -------
    dict
        { user: { "mean_hour": float, "std_hour": float }, ... }
    """
    user_hours: Dict[str, List[int]] = {}

    for e in events:
        user = e.get("user", "unknown")
        ts = e.get("TimeCreated")
        if not ts:
            continue

        dt = parse_windows_time(ts)
        if dt:
            user_hours.setdefault(user, []).append(dt.hour)

    profile: Dict[str, Dict[str, float]] = {}

    for user, hours in user_hours.items():
        if not hours:
            # Reasonable defaults
            mean_h = 12.0
            std_h = 4.0
        else:
            mean_h = float(np.mean(hours))
            std_val = float(np.std(hours)) if len(hours) > 1 else 1.0
            std_h = max(std_val, 1.0)     # avoid zero

        profile[user] = {
            "mean_hour": mean_h,
            "std_hour": std_h,
        }

    return profile


# ============================================================================
#  TRAIN MODEL (Auto-KSelection + Scaler + Metadata)
# ============================================================================

def train_model(events: List[Dict], config, save_path: str):
    """
    Train a KMeans model with MinMax scaling and auto-K selection.

    Steps:
        1) Derive user profile (mean/std hour)
        2) Extract features from events
        3) Fit MinMaxScaler
        4) Try different K values, pick best silhouette score
        5) Save model, scaler, and metadata (cluster_dist, user_profile, etc.)

    Parameters
    ----------
    events : list of dict
        Authentication events.

    config : dict
        Full YAML configuration dictionary.

    save_path : str
        Base filepath for saving model artifacts.

    Returns
    -------
    (model, scaler, meta)
    """
    print(f"[MODEL] Training on {len(events)} events...")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1) User behavioral profile
    user_profile = build_user_profile(events)

    # 2) Feature matrix
    X_raw = np.array(
        [extract_features(e, user_profile) for e in events],
        dtype="float32"
    )

    # 3) Fit scaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)

    # 4) Auto-K
    ks = config.get("modeling", {}).get("candidate_k", [4, 6, 8, 10, 12])
    best_k = None
    best_model = None
    best_score = -1.0

    for k in ks:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X)

            # Reject degenerate solutions (all points same cluster)
            if len(set(labels)) == 1:
                continue

            score = silhouette_score(X, labels)
            print(f"[MODEL] k={k} silhouette={score:.4f}")

            if score > best_score:
                best_score = score
                best_model = km
                best_k = k

        except Exception as exc:
            print(f"[MODEL] k={k} failed: {exc}")
            continue

    # Fallback if all fail
    if best_model is None:
        print("[MODEL] Fallback: k=4 (all Ks collapsed)")
        best_model = KMeans(n_clusters=4, random_state=42, n_init="auto").fit(X)
        best_k = 4

    # 5) Cluster distribution (label counts)
    unique, counts = np.unique(best_model.labels_, return_counts=True)
    cluster_dist = {int(k): int(v) for k, v in zip(unique, counts)}

    # 6) Serialize user profile (JSON-safe)
    safe_user_profile = {
        user: {
            "mean_hour": float(p["mean_hour"]),
            "std_hour": float(p["std_hour"]),
        }
        for user, p in user_profile.items()
    }

    # 7) Metadata
    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "events": len(events),
        "feature_dim": X.shape[1],
        "best_k": best_k,
        "user_profile": safe_user_profile,
        "cluster_dist": cluster_dist,
    }

    # Remove all numpy scalar types before saving
    def sanitize(o):
        if isinstance(o, dict):
            return {str(k): sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        return o

    meta = sanitize(meta)

    # 8) Save artifacts
    joblib.dump(best_model, save_path + ".pkl")
    joblib.dump(scaler, save_path + "_scaler.pkl")
    with open(save_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[MODEL] Saved model to {save_path}")
    return best_model, scaler, meta


# ============================================================================
#  LOAD MODEL (KMeans + Scaler + Metadata)
# ============================================================================

def load_model(path: str):
    """
    Load model, scaler, and metadata.

    Parameters
    ----------
    path : str
        Base model path (without extension).

    Returns
    -------
    (model, scaler, meta)
    """
    model = joblib.load(path + ".pkl")
    scaler = joblib.load(path + "_scaler.pkl")
    with open(path + ".json") as f:
        meta = json.load(f)
    return model, scaler, meta


# ============================================================================
#  PREDICT + ANOMALY SCORE
# ============================================================================

def predict(model_data, events: List[Dict]) -> List[Dict]:
    """
    Run inference on events and compute multi-layer anomaly score.

    Layers:
        1) outlier_score:
            • Distance to KMeans centroid (scaled 0–1)

        2) cluster_rarity:
            • How rarely the user falls into this cluster
            • 1 - freq(user, cluster) / total_user_events

        3) signature_rarity:
            • 1 - P(signature | cluster)

        4) user_hour_score:
            • Z-score deviation of login time
            • normalized so ≥3σ → score ~ 1

        final_anomaly_score:
            0.4*outlier + 0.3*cluster_rarity
          + 0.2*signature_rarity + 0.1*user_hour_score

    This function does **NOT** assign the `behavior_outlier` flag.
    That happens in the pipeline after dynamic thresholding.

    Returns
    -------
    list of dict
        Each enriched event with anomaly fields appended.
    """
    model, scaler, meta = model_data
    user_profile = meta.get("user_profile", {})

    if not events:
        return []

    # --- First pass: Feature extraction & cluster prediction ---
    X_raw = np.array(
        [extract_features(e, user_profile) for e in events],
        dtype="float32"
    )
    X_scaled = scaler.transform(X_raw)

    labels = model.predict(X_scaled)
    distances = model.transform(X_scaled)
    centroid_dist = distances.min(axis=1)

    # Normalize distance to [0,1]
    outlier_score = (centroid_dist - centroid_dist.min()) / (
        (centroid_dist.max() - centroid_dist.min()) + 1e-9
    )

    # Build user-cluster histogram
    user_cluster_hist: Dict[str, Dict[int, int]] = {}
    cluster_sig_hist: Dict[int, Dict[str, int]] = {}

    for e, cid in zip(events, labels):
        cid = int(cid)
        user = e.get("user", "unknown")
        sig = e.get("signature")

        user_cluster_hist.setdefault(user, {})
        user_cluster_hist[user][cid] = user_cluster_hist[user].get(cid, 0) + 1

        cluster_sig_hist.setdefault(cid, {})
        cluster_sig_hist[cid][sig] = cluster_sig_hist[cid].get(sig, 0) + 1

    total_user_events = {u: sum(d.values()) for u, d in user_cluster_hist.items()}

    # Convert signature histograms to probability distributions
    cluster_sig_dist: Dict[int, Dict[str, float]] = {}
    for cid, sig_counts in cluster_sig_hist.items():
        total = float(sum(sig_counts.values())) or 1.0
        cluster_sig_dist[cid] = {sig: cnt / total for sig, cnt in sig_counts.items()}

    # --- Second pass: Anomaly scoring ---
    output: List[Dict] = []

    for e, cid, base_out in zip(events, labels, outlier_score):
        cid = int(cid)
        user = e.get("user", "unknown")

        # cluster_rarity
        u_total = total_user_events.get(user, 1)
        u_c_freq = user_cluster_hist.get(user, {}).get(cid, 0)
        cluster_rarity = 1.0 - (u_c_freq / u_total)

        # signature_rarity
        sig = e.get("signature")
        signature_rarity = 1.0 - cluster_sig_dist.get(cid, {}).get(sig, 0.0)

        # Login hour score
        ts = e.get("TimeCreated")
        dt = parse_windows_time(ts)
        hour = dt.hour if dt else 12

        up = user_profile.get(user, {"mean_hour": 12.0, "std_hour": 4.0})
        mean_h = up["mean_hour"]
        std_h = up["std_hour"] or 1.0

        z = abs(hour - mean_h) / std_h
        user_hour_score = min(z / 3.0, 1.0)

        # Final composite score
        final_score = (
            0.4 * float(base_out)
            + 0.3 * float(cluster_rarity)
            + 0.2 * float(signature_rarity)
            + 0.1 * float(user_hour_score)
        )

        out_evt = dict(e)
        out_evt.update(
            {
                "cluster_id": cid,
                "outlier_score": float(base_out),
                "cluster_rarity": float(cluster_rarity),
                "signature_rarity": float(signature_rarity),
                "user_hour_score": float(user_hour_score),
                "final_anomaly_score": float(final_score),
            }
        )
        output.append(out_evt)

    return output


# ============================================================================
#  DRIFT DETECTION
# ============================================================================

def compute_model_drift(old_dist, new_labels):
    """
    Compare old cluster distribution with new cluster distribution
    using Chi-Square goodness of fit.

    Interpretation:
        • p-value < threshold (e.g., 0.05) → Drift detected
        • p-value >= threshold → No drift

    Parameters
    ----------
    old_dist : dict
        { cluster_id: count }

    new_labels : list or array
        New predicted cluster IDs.

    Returns
    -------
    float
        p-value of drift test. Lower p-value → greater drift.
    """
    old_dist_clean = {int(k): float(v) for k, v in old_dist.items()}

    uniq, cnt = np.unique(new_labels, return_counts=True)
    new_dist = {int(u): float(c) for u, c in zip(uniq, cnt)}

    # Align cluster IDs
    keys = sorted(set(old_dist_clean.keys()) | set(new_dist.keys()))

    old_arr = np.array([old_dist_clean.get(k, 1e-6) for k in keys], dtype=float)
    new_arr = np.array([new_dist.get(k, 1e-6) for k in keys], dtype=float)

    # Avoid divide-by-zero
    if old_arr.sum() == 0 or new_arr.sum() == 0:
        return 1.0

    old_arr = old_arr / old_arr.sum()
    new_arr = new_arr / new_arr.sum()

    try:
        chi2, p = chisquare(new_arr, f_exp=old_arr)
        return p
    except Exception as e:
        print("[DRIFT] WARNING:", e, "→ fallback p=1.0")
        return 1.0