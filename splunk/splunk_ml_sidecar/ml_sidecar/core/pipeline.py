#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — Full Auto-Pipeline for Training, Drift Detection, Inference,
              Anomaly Scoring, and KVStore Export.

This module orchestrates the full ML pipeline of the Splunk ML Sidecar.

Major Responsibilities:
- Load configuration and ingestion rules
- Ingest authentication events
- Train or load KMeans model + scaler
- Perform model drift detection (Chi-Square)
- Run inference + anomaly scoring
- Apply dynamic anomaly thresholds
- Build behavioral profiles (cluster / user / events)
- Export results into Splunk KVStore for dashboards

The pipeline is stateful: it persists model artifacts and updates them over time.
"""

import os
import json
from datetime import datetime
import numpy as np

from .config_loader import load_settings
from .ingestion import load_splunk_events
from .utils import ensure_dir
from .model import (
    model_exists,
    load_model,
    train_model,
    predict,
    compute_model_drift,
)
from .profiles import (
    build_cluster_profiles,
    build_user_profiles,
    build_event_records,
    build_outlier_event_records,
)
from .kvstore import write_kvstore_collection


# ============================================================================
# AUTO PIPELINE ENTRYPOINT
# ============================================================================
def run_auto_pipeline():
    """
    Main entrypoint for the full ML pipeline.

    This function ties together:
        ingestion → training → drift detection →
        inference → profiling → KVStore export

    It is designed to be called by a scheduler or modular input.
    """
    print("\n=== ML Sidecar Auto-Pipeline ===")

    # ----------------------------------------------------------------------
    # 1) LOAD CONFIGURATION
    # ----------------------------------------------------------------------
    cfg = load_settings()  # read YAML config settings from config/settings.yaml

    # ----------------------------------------------------------------------
    # 2) INGEST AUTHENTICATION EVENTS FROM SPLUNK
    # ----------------------------------------------------------------------
    print("[AUTO] Fetching events from Splunk…")
    events = load_splunk_events(cfg["ingestion"])
    total = len(events)
    print(f"[AUTO] Loaded {total} authentication events")

    # If there is no data, pipeline should exit gracefully
    if total == 0:
        print("[AUTO] No events found → exiting")
        return []

    # ----------------------------------------------------------------------
    # 3) INITIALIZE MODEL PATHS
    # ----------------------------------------------------------------------
    model_dir = cfg["general"]["model_dir"]
    model_name = cfg["general"]["model_name"]
    model_path = os.path.join(model_dir, model_name)

    ensure_dir(model_dir)   # create model directory if not exists

    # modeling hyperparameters from config
    modeling_cfg = cfg.get("modeling", {})
    drift_threshold = modeling_cfg.get("drift_threshold", 0.05)
    outlier_percentile = modeling_cfg.get("outlier_percentile", 0.99)

    # ensure percentile is sane
    if not (0.0 < outlier_percentile < 1.0):
        print(f"[AUTO] Invalid outlier percentile '{outlier_percentile}', defaulting to 0.99")
        outlier_percentile = 0.99

    # ----------------------------------------------------------------------
    # 4) TRAIN OR LOAD EXISTING MODEL
    # ----------------------------------------------------------------------
    if not model_exists(model_path):
        # No prior model → first-time training
        print("[AUTO] No model found → training initial model…")
        model, scaler, meta = train_model(events, cfg, model_path)

    else:
        # Load previously trained model + metadata
        print("[AUTO] Existing model found → loading from disk…")
        model, scaler, meta = load_model(model_path)

        # ------------------------------------------------------------------
        # 5) DRIFT DETECTION (Chi-square)
        # ------------------------------------------------------------------
        print("[AUTO] Performing drift detection…")

        # Historical user profile is required for feature extraction
        user_profile = meta.get("user_profile", {})

        # Import here to avoid circular dependency
        from .features import extract_features

        # Convert events into feature matrix for drift check
        X_raw = np.array(
            [extract_features(e, user_profile) for e in events],
            dtype="float32"
        )
        X_scaled = scaler.transform(X_raw)

        # Predict cluster labels for new batch
        new_labels = model.predict(X_scaled)

        # If model has no reference distribution, drift cannot be computed
        if "cluster_dist" not in meta:
            print("[AUTO] No historical cluster distribution → skipping drift detection")

        else:
            # p-value < threshold → drift occurred → retrain
            p = compute_model_drift(meta["cluster_dist"], new_labels)
            print(f"[AUTO] Drift p-value = {p:.6f} (threshold={drift_threshold})")

            if p < drift_threshold:
                print("[AUTO] Drift detected → retraining model…")
                model, scaler, meta = train_model(events, cfg, model_path)
            else:
                print("[AUTO] No drift detected → keeping current model")

    # ----------------------------------------------------------------------
    # 6) INFERENCE (CLUSTER + ANOMALY SCORING)
    # ----------------------------------------------------------------------
    print("[AUTO] Running inference and anomaly scoring…")
    enriched = predict((model, scaler, meta), events)

    # ----------------------------------------------------------------------
    # 7) PER-USER NORMALIZATION + THRESHOLDING (NEW LOGIC)
    # ----------------------------------------------------------------------
    print("[AUTO] Applying per-user normalization…")

    # Build user → score list
    user_scores = {}
    for e in enriched:
        user = e.get("user", "unknown")
        score = float(e.get("final_anomaly_score", 0.0))
        user_scores.setdefault(user, []).append(score)

    # Compute per-user mean & std
    user_stats = {}
    for user, scores in user_scores.items():
        mean_u = float(np.mean(scores))
        std_u = float(np.std(scores))
        std_u = std_u if std_u > 0.1 else 0.1  # prevent division-by-zero
        user_stats[user] = (mean_u, std_u)

    # Compute per-user z-score and sigmoid normalizations
    for e in enriched:
        user = e.get("user", "unknown")
        score = float(e.get("final_anomaly_score", 0.0))

        mean_u, std_u = user_stats.get(user, (0.5, 0.2))
        z = (score - mean_u) / std_u

        # Sigmoid for 0–1 normalization
        norm_score = 1.0 / (1.0 + np.exp(-z))

        e["per_user_zscore"] = float(z)
        e["per_user_norm_score"] = float(norm_score)

    # Compute per-user outlier thresholds (99th percentile)
    print("[AUTO] Computing per-user thresholds…")

    user_thresholds = {}
    for user, scores in user_scores.items():
        t = float(np.quantile(scores, outlier_percentile))
        user_thresholds[user] = t

    a = []
    # Apply final outlier decision
    for e in enriched:
        user = e.get("user", "unknown")
        score = float(e.get("final_anomaly_score", 0.0))
        threshold_u = user_thresholds.get(user, 1.0)

        e["per_user_threshold"] = threshold_u
        e["behavior_outlier"] = int(score >= threshold_u)

    print("[AUTO] Per-user anomaly detection complete.")

    # ----------------------------------------------------------------------
    # 8) DYNAMIC OUTLIER THRESHOLDING
    # ----------------------------------------------------------------------
    # Collect final_anomaly_score values for percentile thresholding
    scores = [e.get("final_anomaly_score", 0.0) for e in enriched]

    # Compute threshold based on percentile
    threshold = float(np.quantile(scores, outlier_percentile)) if scores else 1.0
    print(f"[AUTO] Behavior threshold (percentile={outlier_percentile}) = {threshold:.4f}")

    # Apply outlier flag
    for e in enriched:
        e["behavior_outlier"] = int(e["final_anomaly_score"] >= threshold)

    # ----------------------------------------------------------------------
    # 9) PROFILE GENERATION (CLUSTER / USER / EVENT)
    # ----------------------------------------------------------------------
    print("[AUTO] Building behavioral profiles…")
    cluster_profiles = build_cluster_profiles(enriched)
    user_profiles = build_user_profiles(enriched)
    event_records = build_event_records(enriched)
    outlier_event_records = build_outlier_event_records(enriched)

    # Show example event for debugging
    if event_records:
        print("\n[DEBUG] Sample enriched event:")
        print(json.dumps(event_records[0], indent=2))

    # ----------------------------------------------------------------------
    # 10) EXPORT TO SPLUNK KVSTORE
    # ----------------------------------------------------------------------
    kv_cfg = cfg["output"].get("splunk_kvstore", {})

    # KVStore export can be turned off in config
    if not kv_cfg.get("enabled", False):
        print("[AUTO] KVStore export disabled → pipeline complete")
        return enriched

    base_url = kv_cfg["base_url"]
    token = kv_cfg["auth_token"]
    app = kv_cfg.get("app", "ml_sidecar_app")

    print("[AUTO] Exporting profiles to KVStore…")

    # Write 3 profile collections
    write_kvstore_collection(cluster_profiles, base_url, token, app, "auth_cluster_profiles")
    write_kvstore_collection(user_profiles, base_url, token, app, "auth_user_profiles")
    write_kvstore_collection(event_records, base_url, token, app, "auth_events")
    # write_kvstore_collection(outlier_event_records, base_url, token, app, "auth_outlier_events")

    print("=== Auto-Pipeline Completed Successfully ===")
    return enriched