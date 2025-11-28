#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — Full Auto-Pipeline for Training, Drift Detection, Inference,
              Anomaly Scoring, and KVStore Export

This module implements the "Auto Pipeline" of the Splunk ML Sidecar.

The pipeline performs:
1) Splunk ingestion
2) Model load or training
3) Drift detection using Chi-Square distribution shift
4) Full inference on all events
5) Composite anomaly scoring with dynamic thresholds
6) Profile generation (cluster/user/event)
7) KVStore export (4 collections)

Notes
-----
• This module is **stateful** — it updates model files on disk.
• All interactions with Splunk (REST/KVStore) are delegated to `ingestion.py` and `kvstore.py`.
• The pipeline is designed for cron, scheduled tasks, and long-running systems.
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
)
from .kvstore import write_kvstore_collection
from .features import extract_features


# ============================================================================
# AUTO PIPELINE (MAIN ENTRYPOINT)
# ============================================================================

def run_auto_pipeline():
    """
    Run the full automated ML pipeline.

    Steps:
        1) Load settings
        2) Fetch Splunk authentication events
        3) Train initial model OR load existing model
        4) Perform drift detection (Chi-square)
        5) Retrain if drift exceeds threshold
        6) Inference: generate cluster IDs & anomaly scores
        7) Apply dynamic threshold and mark outliers
        8) Build cluster/user/event profiles
        9) Write results to Splunk KVStore (4 collections)

    Returns:
        list of enriched events (dicts)
    """
    print("\n=== ML Sidecar Auto-Pipeline ===")

    cfg = load_settings()

    # ----------------------------------------------------------------------
    # 1) INGESTION
    # ----------------------------------------------------------------------
    print("[AUTO] Fetching events…")
    events = load_splunk_events(cfg["ingestion"])
    total = len(events)
    print(f"[AUTO] Loaded {total} events from Splunk")

    if total == 0:
        print("[AUTO] No events → exit")
        return []

    # ----------------------------------------------------------------------
    # 2) MODEL PATH SETUP
    # ----------------------------------------------------------------------
    model_dir = cfg["general"]["model_dir"]
    model_name = cfg["general"]["model_name"]
    model_path = os.path.join(model_dir, model_name)
    ensure_dir(model_dir)

    modeling_cfg = cfg.get("modeling", {})
    drift_threshold = modeling_cfg.get("drift_threshold", 0.05)
    outlier_percentile = modeling_cfg.get("outlier_percentile", 0.99)

    # Validate percentile
    if not (0.0 < outlier_percentile < 1.0):
        print(f"[AUTO] Invalid outlier_percentile={outlier_percentile}, using 0.99")
        outlier_percentile = 0.99

    # ----------------------------------------------------------------------
    # 3) TRAIN OR LOAD MODEL
    # ----------------------------------------------------------------------
    if not model_exists(model_path):
        print("[AUTO] No existing model → Training initial model on all events")
        model, scaler, meta = train_model(events, cfg, model_path)
    else:
        print("[AUTO] Loading existing model from disk")
        model, scaler, meta = load_model(model_path)

        # ------------------------------------------------------------------
        # 4) DRIFT DETECTION
        # ------------------------------------------------------------------
        print("[AUTO] Checking drift…")

        # Use historical user profile from model metadata if available
        user_profile = meta.get("user_profile", {})

        # Prepare feature matrix for drift check
        X_raw = np.array(
            [extract_features(e, user_profile) for e in events],
            dtype="float32"
        )
        X_scaled = scaler.transform(X_raw)
        new_labels = model.predict(X_scaled)

        if "cluster_dist" not in meta:
            print("[AUTO] No previous cluster distribution → no drift check applied")
        else:
            p_value = compute_model_drift(meta["cluster_dist"], new_labels)
            print(f"[AUTO] Drift p-value = {p_value:.6f} (threshold={drift_threshold})")

            # p-value < threshold  → drift detected → retrain
            if p_value < drift_threshold:
                print("[AUTO] Drift detected → retraining model")
                model, scaler, meta = train_model(events, cfg, model_path)
            else:
                print("[AUTO] No significant drift → keeping existing model")

    # ----------------------------------------------------------------------
    # 5) INFERENCE — CLUSTER PREDICTION + ANOMALY SCORING
    # ----------------------------------------------------------------------
    print("[AUTO] Running full inference…")
    enriched = predict((model, scaler, meta), events)

    # ----------------------------------------------------------------------
    # 6) DYNAMIC THRESHOLD — OUTLIER LABELING
    # ----------------------------------------------------------------------
    scores = [e.get("final_anomaly_score", 0.0) for e in enriched]

    if scores:
        threshold = float(np.quantile(scores, outlier_percentile))
    else:
        threshold = 1.0

    print(f"[AUTO] Behavior threshold (p={outlier_percentile}) = {threshold:.4f}")

    for e in enriched:
        e["behavior_outlier"] = int(e.get("final_anomaly_score", 0.0) >= threshold)

    # ----------------------------------------------------------------------
    # 7) PROFILE GENERATION
    # ----------------------------------------------------------------------
    print("[AUTO] Building profiles…")
    cluster_profiles = build_cluster_profiles(enriched)
    user_profiles = build_user_profiles(enriched)
    event_records = build_event_records(enriched)

    if event_records:
        print("\n[DEBUG] Sample event:")
        print(json.dumps(event_records[0], indent=2))
    else:
        print("[DEBUG] No event records generated")

    # ----------------------------------------------------------------------
    # 8) KVSTORE EXPORT LOGIC
    # ----------------------------------------------------------------------
    kv_cfg = cfg["output"].get("splunk_kvstore", {})

    if not kv_cfg.get("enabled", False):
        print("[AUTO] KVStore export disabled → skipping")
        print("=== Auto-Pipeline Complete ===")
        return enriched

    base_url = kv_cfg["base_url"]
    token = kv_cfg["auth_token"]
    app = kv_cfg.get("app", "ml_sidecar_app")

    # Write 3 profile collections
    print("[AUTO] Writing cluster profiles…")
    write_kvstore_collection(cluster_profiles, base_url, token, app, "auth_cluster_profiles")

    print("[AUTO] Writing user profiles…")
    write_kvstore_collection(user_profiles, base_url, token, app, "auth_user_profiles")

    print("[AUTO] Writing event profiles…")
    write_kvstore_collection(event_records, base_url, token, app, "auth_events")

    # Daily threshold record
    today = datetime.utcnow().date().isoformat()
    threshold_doc = [{
        "_key": today,
        "date": today,
        "threshold": float(threshold),
        "percentile": float(outlier_percentile),
        "event_count": len(scores),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }]

    print("[AUTO] Writing daily threshold document…")
    write_kvstore_collection(threshold_doc, base_url, token, app, "auth_thresholds")

    print("=== Auto-Pipeline Completed ===")
    return enriched