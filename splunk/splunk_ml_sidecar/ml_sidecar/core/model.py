#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py — KMeans model training, loading, inference, and drift detection logic.

This module provides the core ML logic for the Splunk ML Sidecar:

- Builds user behavior profiles (mean/std login hour)
- Extracts feature vectors from events
- Trains a KMeans model (auto-selects best K via silhouette score)
- Computes multi-layer anomaly scores during prediction
- Detects model drift using Chi-square distribution shift

Used by:
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
#  CHECK IF MODEL EXISTS
# ============================================================================

def model_exists(path: str) -> bool:
    """
    Check whether the trained model file (<path>.pkl) already exists.

    Only the KMeans pickle is required for detecting existence;
    scaler and metadata are expected to exist alongside it.
    """
    return os.path.exists(path + ".pkl")


# ============================================================================
#  BUILD USER PROFILE (MEAN/STDEV LOGIN HOUR)
# ============================================================================

def build_user_profile(events: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Build per-user temporal behavioral profile.

    For each user:
        - mean_hour: average login hour
        - std_hour: variability in login hour

    These values help detect temporal anomalies during prediction.
    """
    user_hours: Dict[str, List[int]] = {}

    # Collect login hours for each user
    for e in events:
        user = e.get("user", "unknown")
        ts = e.get("TimeCreated")
        dt = parse_windows_time(ts)
        if dt:
            user_hours.setdefault(user, []).append(dt.hour)

    # Compute statistics
    profile = {}

    for user, hours in user_hours.items():
        if hours:
            # Standard temporal behavior from real timestamps
            mean_h = float(np.mean(hours))
            std_val = float(np.std(hours)) if len(hours) > 1 else 1.0
        else:
            # Fallback when user has no timestamped events
            mean_h = 12.0
            std_val = 4.0

        # std_hour must never be zero (avoid division-by-zero later)
        profile[user] = {
            "mean_hour": mean_h,
            "std_hour": max(std_val, 1.0)
        }

    return profile


# ============================================================================
#  TRAIN KMEANS MODEL WITH AUTO-K
# ============================================================================

def train_model(events: List[Dict], config, save_path: str):
    """
    Train a new KMeans model with MinMaxScaler and auto-K selection.

    Steps:
        1. Build user behavioral profile
        2. Extract features into a numeric matrix
        3. Fit MinMaxScaler (normalize 0–1)
        4. Try multiple K values → pick best silhouette score
        5. Save model, scaler, and metadata to disk
    """
    print(f"[MODEL] Training on {len(events)} events...")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. User temporal behavior
    user_profile = build_user_profile(events)

    # 2. Event → feature vector (numeric)
    X_raw = np.array([extract_features(e, user_profile) for e in events], dtype="float32")

    # 3. Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)

    # 4. Auto-K selection (try multiple K values)
    ks = config.get("modeling", {}).get("candidate_k", [4, 6, 8, 10, 12])
    best_model, best_k, best_score = None, None, -1

    for k in ks:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X)

            # Degenerate case: model assigns all points to same cluster
            if len(set(labels)) == 1:
                continue

            score = silhouette_score(X, labels)
            print(f"[MODEL] k={k}, silhouette={score:.4f}")

            # Keep model with highest silhouette score
            if score > best_score:
                best_model = km
                best_k = k
                best_score = score

        except Exception as exc:
            print(f"[MODEL] k={k} failed: {exc}")

    # Fallback if all K attempts fail
    if best_model is None:
        print("[MODEL] Fallback: k=4")
        best_model = KMeans(n_clusters=4, random_state=42, n_init="auto").fit(X)
        best_k = 4

    # 5. Baseline cluster distribution (for drift detection later)
    unique, counts = np.unique(best_model.labels_, return_counts=True)
    cluster_dist = {int(k): int(v) for k, v in zip(unique, counts)}

    # Prepare metadata
    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "events": len(events),
        "feature_dim": X.shape[1],
        "best_k": best_k,

        # user profile saved for inference use
        "user_profile": {
            user: {
                "mean_hour": float(p["mean_hour"]),
                "std_hour": float(p["std_hour"])
            }
            for user, p in user_profile.items()
        },

        # cluster distribution for drift detection
        "cluster_dist": cluster_dist
    }

    # Save to disk
    joblib.dump(best_model, save_path + ".pkl")
    joblib.dump(scaler, save_path + "_scaler.pkl")

    with open(save_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[MODEL] Saved model to {save_path}")
    return best_model, scaler, meta


# ============================================================================
#  LOAD MODEL FROM DISK
# ============================================================================

def load_model(path: str):
    """
    Load a trained model, scaler, and metadata from disk.
    """
    model = joblib.load(path + ".pkl")
    scaler = joblib.load(path + "_scaler.pkl")
    with open(path + ".json") as f:
        meta = json.load(f)
    return model, scaler, meta


# ============================================================================
#  INFERENCE + MULTI-LAYER ANOMALY SCORING
# ============================================================================

def predict(model_data, events: List[Dict]) -> List[Dict]:
    """
    Run inference on events and compute a multi-layer anomaly score.

    Layers:
        1. outlier_score        → distance from cluster centroid
        2. cluster_rarity       → unusual cluster assignment for user
        3. signature_rarity     → rare signature_id within cluster
        4. user_hour_score      → deviation from user's normal login hour

    final_anomaly_score = weighted sum of layers

    Returns:
        list of enriched events
    """
    model, scaler, meta = model_data
    user_profile = meta.get("user_profile", {})

    if not events:
        return []

    # ---------------- FIRST PASS: embed + cluster ----------------
    X_raw = np.array([extract_features(e, user_profile) for e in events], dtype="float32")
    X = scaler.transform(X_raw)

    # Assign clusters
    labels = model.predict(X)

    # Distance to each centroid → take minimum distance
    distances = model.transform(X)
    centroid_dist = distances.min(axis=1)

    # Normalize distances to 0–1
    outlier_score = (centroid_dist - centroid_dist.min()) / (
        (centroid_dist.max() - centroid_dist.min()) + 1e-9
    )

    # Build histograms
    user_cluster_hist = {}      # user → {cluster: count}
    cluster_sig_hist = {}       # cluster → {signature_id: count}

    for e, cid in zip(events, labels):
        cid = int(cid)
        user = e.get("user", "unknown")
        sig_id = str(e.get("signature_id") or "0")

        # Count cluster usage per user
        user_cluster_hist.setdefault(user, {})
        user_cluster_hist[user][cid] = user_cluster_hist[user].get(cid, 0) + 1

        # Count signature usage per cluster
        cluster_sig_hist.setdefault(cid, {})
        cluster_sig_hist[cid][sig_id] = cluster_sig_hist[cid].get(sig_id, 0) + 1

    # Compute total per user (needed for rarity score)
    total_user_events = {u: sum(c.values()) for u, c in user_cluster_hist.items()}

    # Convert signature histograms to probability distributions
    cluster_sig_dist = {}
    for cid, sig_counts in cluster_sig_hist.items():
        total = float(sum(sig_counts.values())) or 1.0
        cluster_sig_dist[cid] = {sig: cnt / total for sig, cnt in sig_counts.items()}

    # ---------------- SECOND PASS: scoring ----------------
    enriched = []

    for e, cid, base_out in zip(events, labels, outlier_score):
        cid = int(cid)
        user = e.get("user", "unknown")
        sig_id = str(e.get("signature_id") or "0")

        # (2) How unusual is this cluster for the specific user?
        u_total = total_user_events.get(user, 1)
        u_freq = user_cluster_hist.get(user, {}).get(cid, 0)
        cluster_rarity = 1.0 - (u_freq / u_total)

        # (3) How rare is this signature_id inside the cluster?
        signature_rarity = 1.0 - cluster_sig_dist.get(cid, {}).get(sig_id, 0.0)

        # (4) Temporal deviation (Z-score of login hour)
        dt = parse_windows_time(e.get("TimeCreated"))
        hour = dt.hour if dt else 12
        up = user_profile.get(user, {"mean_hour": 12.0, "std_hour": 4.0})

        mean_h, std_h = up["mean_hour"], up["std_hour"]
        z = abs(hour - mean_h) / std_h
        user_hour_score = min(z / 3.0, 1.0)  # cap at 1.0

        # Weighted composite anomaly score
        final_score = (
            0.4 * float(base_out) +
            0.3 * float(cluster_rarity) +
            0.2 * float(signature_rarity) +
            0.1 * float(user_hour_score)
        )

        # Produce enriched event copy
        out_evt = dict(e)
        out_evt.update({
            "cluster_id": cid,
            "outlier_score": float(base_out),
            "cluster_rarity": float(cluster_rarity),
            "signature_rarity": float(signature_rarity),
            "user_hour_score": float(user_hour_score),
            "final_anomaly_score": float(final_score),
        })

        enriched.append(out_evt)

    return enriched


# ============================================================================
#  DRIFT DETECTION USING CHI-SQUARE
# ============================================================================

def compute_model_drift(old_dist, new_labels):
    """
    Compare the original cluster distribution (from training)
    with distribution of current events using Chi-square test.

    Interpretation:
        p < threshold → DRIFT detected (distribution changed significantly)
        p ≥ threshold → NO drift

    Returns:
        float — p-value of the goodness-of-fit test
    """
    old_dist = {int(k): float(v) for k, v in old_dist.items()}

    # Build new distribution
    uniq, cnt = np.unique(new_labels, return_counts=True)
    new_dist = {int(i): float(c) for i, c in zip(uniq, cnt)}

    # Align keys
    keys = sorted(set(old_dist.keys()) | set(new_dist.keys()))

    # Convert to comparable arrays
    old_arr = np.array([old_dist.get(k, 1e-6) for k in keys], dtype=float)
    new_arr = np.array([new_dist.get(k, 1e-6) for k in keys], dtype=float)

    # Avoid zero-sum distributions
    if old_arr.sum() == 0 or new_arr.sum() == 0:
        return 1.0

    # Normalize to probability distributions
    old_arr /= old_arr.sum()
    new_arr /= new_arr.sum()

    # Chi-square goodness of fit
    try:
        chi2, p = chisquare(new_arr, f_exp=old_arr)
        return p
    except Exception:
        # Any error → treat as no drift detected
        return 1.0