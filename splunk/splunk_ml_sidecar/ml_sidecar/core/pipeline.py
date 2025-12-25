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
    # 7) PER-USER NORMALIZATION + THRESHOLDING
    # ----------------------------------------------------------------------
    print("[AUTO] Applying per-user normalization…")

    # 1) Collect per-user raw final scores
    user_raw_scores = {}
    for e in enriched:
        user = e.get("user", "unknown")
        user_raw_scores.setdefault(user, []).append(
            float(e.get("final_anomaly_score", 0.0))
        )

    # 2) Compute per-user mean/std for z-score
    user_stats = {}
    for user, scores in user_raw_scores.items():
        mean_u = float(np.mean(scores))
        std_u = float(np.std(scores))
        std_u = std_u if std_u > 0.1 else 0.1  # avoid division-by-zero / tiny std
        user_stats[user] = (mean_u, std_u)

    # 3) Compute per-event per_user_zscore and per_user_norm_score (sigmoid)
    user_norm_scores = {}  # store per-user norm_score list (for threshold)
    for e in enriched:
        user = e.get("user", "unknown")
        score = float(e.get("final_anomaly_score", 0.0))

        mean_u, std_u = user_stats.get(user, (0.5, 0.2))
        z = (score - mean_u) / std_u

        # Sigmoid maps (-inf, +inf) => (0,1)
        norm_score = 1.0 / (1.0 + np.exp(-z))

        e["per_user_zscore"] = float(z)
        e["per_user_norm_score"] = float(norm_score)

        user_norm_scores.setdefault(user, []).append(float(norm_score))

    # 4) Compute per-user thresholds on *normalized* scores (recommended)
    print("[AUTO] Computing per-user thresholds (on per_user_norm_score)…")
    user_thresholds_norm = {}
    for user, nscores in user_norm_scores.items():
        t = float(np.quantile(nscores, outlier_percentile))
        user_thresholds_norm[user] = t

    # 5) Apply per-user outlier decision (normalized score)
    for e in enriched:
        user = e.get("user", "unknown")
        norm_score = float(e.get("per_user_norm_score", 0.0))
        threshold_norm = float(user_thresholds_norm.get(user, 1.0))

        e["per_user_threshold_norm"] = threshold_norm
        e["behavior_outlier_user"] = int(norm_score >= threshold_norm)

    print("[AUTO] Per-user anomaly detection complete.")

    # ----------------------------------------------------------------------
    # 8) DYNAMIC OUTLIER THRESHOLDING
    # ----------------------------------------------------------------------
    # Collect final_anomaly_score values for percentile thresholding

    # Compute global threshold
    scores = [e.get("final_anomaly_score", 0.0) for e in enriched]
    global_threshold = float(np.quantile(scores, outlier_percentile)) if scores else 1.0

    # ----------------------------------------------------------------------
    # 8.5) CLUSTER-BASED OUTLIER THRESHOLDING
    # ----------------------------------------------------------------------
    from collections import defaultdict

    cluster_scores = defaultdict(list)

    for e in enriched:
        cluster_scores[e["cluster_id"]].append(
            float(e.get("final_anomaly_score", 0.0))
        )

    MIN_CLUSTER_EVENTS = modeling_cfg.get("min_cluster_events", 30)

    cluster_thresholds = {
        cid: float(np.quantile(scores, outlier_percentile))
        for cid, scores in cluster_scores.items()
        if len(scores) >= MIN_CLUSTER_EVENTS
    }

    print(f"[AUTO] Computed {len(cluster_thresholds)} cluster-specific thresholds")

    # Apply cluster outlier decision
    for e in enriched:
        cid = e.get("cluster_id")
        threshold_c = cluster_thresholds.get(cid, global_threshold)

        e["cluster_threshold"] = threshold_c
        e["behavior_outlier_cluster"] = int(
            e.get("final_anomaly_score", 0.0) >= threshold_c
        )


    print(f"[AUTO] Global behavior threshold (p={outlier_percentile}) = {global_threshold:.4f}")

    # Apply global outlier decision
    for e in enriched:
        e["behavior_outlier_global"] = int(
            e.get("final_anomaly_score", 0.0) >= global_threshold
        )

    VALID_MODES = {"user", "cluster", "global", "combined"}

    # Final outlier decision (user/global/combined)
    outlier_mode = modeling_cfg.get("outlier_mode", "combined").lower()
    combined_mode = modeling_cfg.get("combined_logic", "and").lower()

    if outlier_mode not in VALID_MODES:
        raise ValueError(f"Invalid outlier_mode: {outlier_mode}")
    
    elif outlier_mode == "combined":
        threshold_cfg = modeling_cfg.get("thresholds", {})

        use_user = threshold_cfg.get("enable_user", True)
        use_cluster = threshold_cfg.get("enable_cluster", True)
        use_global = threshold_cfg.get("enable_global", True)

    for e in enriched:

        # ------------------------------
        # Non-combined modes (direct)
        # ------------------------------
        if outlier_mode != "combined":
            e["behavior_outlier"] = int(e.get(f"behavior_outlier_{outlier_mode}", 0))
            e["outlier_decision_source"] = outlier_mode
            continue

        # ------------------------------
        # Combined mode
        # ------------------------------
        decisions = []

        if use_user:
            decisions.append(int(e.get("behavior_outlier_user", 0)))

        if use_cluster:
            decisions.append(int(e.get("behavior_outlier_cluster", 0)))

        if use_global:
            decisions.append(int(e.get("behavior_outlier_global", 0)))

        # Safety: no threshold enabled → never alert
        if not decisions:
            e["behavior_outlier"] = 0
            e["outlier_decision_source"] = "combined_none"
            continue

        if combined_mode == "or":
            e["behavior_outlier"] = int(any(decisions))
            e["outlier_decision_source"] = "combined_or"
        else:
            e["behavior_outlier"] = int(all(decisions))
            e["outlier_decision_source"] = "combined_and"


    # Logging
    if outlier_mode == "combined":
        print(
            f"[AUTO] Outlier mode = combined ({combined_mode}) | "
            f"user={use_user}, cluster={use_cluster}, global={use_global}"
        )
    else:
        print(f"[AUTO] Outlier mode = {outlier_mode}")


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
    # 10) THRESHOLD INFORMATION (GLOBAL & USER)
    # ----------------------------------------------------------------------

    today = datetime.utcnow().date().isoformat()

    # Build per-user threshold docs (normalized threshold)
    user_threshold_docs = []
    for user, t in user_thresholds_norm.items():
        user_threshold_docs.append({
            "_key": f"{today}_{user}",
            "date": today,
            "threshold_global": float(global_threshold),
            "global_event_count": int(len(scores)),
            "outlier_decision_source": e["outlier_decision_source"],
            "user": user,
            "threshold_user_norm": float(t),
            "percentile": float(outlier_percentile),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        })

    # ----------------------------------------------------------------------
    # 11) EXPORT TO SPLUNK KVSTORE
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

    # Per-user thresholds with global threshold (daily, one doc per user)
    write_kvstore_collection(user_threshold_docs, base_url, token, app, "auth_user_thresholds")

    # Optional: export only outlier events collection
    if modeling_cfg.get("export_outlier_events", False):
        write_kvstore_collection(outlier_event_records, base_url, token, app, "auth_outlier_events")


    print("=== Auto-Pipeline Completed Successfully ===")
    return enriched