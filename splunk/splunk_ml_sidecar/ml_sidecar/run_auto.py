#!/usr/bin/env python3
import os
import json
from datetime import datetime

import numpy as np

from ml_sidecar.config_loader import load_settings
from ml_sidecar.ingestion import load_splunk_events
from ml_sidecar.utils import ensure_dir
from ml_sidecar.model import (
    model_exists,
    load_model,
    train_model,
    predict,
    compute_model_drift,
)
from ml_sidecar.profiles import (
    build_cluster_profiles,
    build_user_profiles,
    build_event_records,
)
from ml_sidecar.kvstore import write_kvstore_collection
from ml_sidecar.features import extract_features


def run_auto_pipeline():
    print("\n=== ML Sidecar Auto-Pipeline ===")
    cfg = load_settings()

    # -------------------------------------------------
    # 1) Events from Splunk
    # -------------------------------------------------
    print("[AUTO] Fetching events…")
    events = load_splunk_events(cfg["ingestion"])
    total = len(events)
    print(f"[AUTO] Loaded {total} events")

    if total == 0:
        print("[AUTO] No events → exit")
        return []

    # -------------------------------------------------
    # 2) Model path
    # -------------------------------------------------
    model_dir = cfg["general"]["model_dir"]
    model_name = cfg["general"]["model_name"]
    model_path = os.path.join(model_dir, model_name)
    ensure_dir(model_dir)

    modeling_cfg = cfg.get("modeling", {})
    drift_threshold = modeling_cfg.get("drift_threshold", 0.05)
    outlier_percentile = modeling_cfg.get("outlier_percentile", 0.99)
    if not (0.0 < outlier_percentile < 1.0):
        outlier_percentile = 0.99

    # -------------------------------------------------
    # 3) Train model if needed
    # -------------------------------------------------
    if not model_exists(model_path):
        print("[AUTO] No model → train initial model on all events")
        model, scaler, meta = train_model(events, cfg, model_path)
    else:
        print("[AUTO] Loading existing model")
        model, scaler, meta = load_model(model_path)

        # -------------------------------------------------
        # 4) Drift detection
        # -------------------------------------------------
        print("[AUTO] Checking drift…")
        user_profile = meta.get("user_profile", {})

        X_raw = np.array(
            [extract_features(e, user_profile) for e in events],
            dtype="float32",
        )
        X_scaled = scaler.transform(X_raw)
        new_labels = model.predict(X_scaled)

        if "cluster_dist" not in meta:
            print("[AUTO] No previous cluster_dist → assume no drift")
        else:
            p = compute_model_drift(meta["cluster_dist"], new_labels)
            print(f"[AUTO] Drift p-value = {p:.6f} (threshold={drift_threshold})")

            if p < drift_threshold:
                print("[AUTO] Drift detected → retrain on current window")
                model, scaler, meta = train_model(events, cfg, model_path)
            else:
                print("[AUTO] No significant drift → keep existing model")

    # -------------------------------------------------
    # 5) Full inference with composite anomaly score
    # -------------------------------------------------
    print("[AUTO] Running full inference…")
    enriched = predict((model, scaler, meta), events)

    # -------------------------------------------------
    # 6) Dynamic threshold & behavior_outlier flag
    # -------------------------------------------------
    scores = [e.get("final_anomaly_score", 0.0) for e in enriched]
    if scores:
        threshold = float(np.quantile(scores, outlier_percentile))
    else:
        threshold = 1.0

    print(f"[AUTO] Behavior threshold (p={outlier_percentile}) = {threshold:.4f}")

    for e in enriched:
        score = float(e.get("final_anomaly_score", 0.0))
        e["behavior_outlier"] = int(score >= threshold)

    # -------------------------------------------------
    # 7) Write enriched events to file (debug/archive)
    # -------------------------------------------------
    out_path = cfg["output"]["file"]["path"]
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        for e in enriched:
            f.write(json.dumps(e) + "\n")
    print(f"[AUTO] Saved inference to {out_path}")

    # -------------------------------------------------
    # 8) Build profiles (cluster / user / event)
    # -------------------------------------------------
    print("[AUTO] Building profiles…")
    cluster_profiles = build_cluster_profiles(enriched)
    user_profiles = build_user_profiles(enriched)
    event_records = build_event_records(enriched)

    # Debug: first event record
    if event_records:
        print("\n[DEBUG] Sample event record:")
        print(json.dumps(event_records[0], indent=2))
    else:
        print("[DEBUG] No event records built!")

    # -------------------------------------------------
    # 9) Write to KVStore (3 collections + thresholds)
    # -------------------------------------------------
    kv_cfg = cfg["output"].get("splunk_kvstore", {})
    if not kv_cfg.get("enabled", False):
        print("[AUTO] KVStore export disabled in config.")
        print("=== Auto-Pipeline Completed (KVStore disabled) ===")
        return enriched

    base_url = kv_cfg["base_url"]
    token = kv_cfg["auth_token"]
    app = kv_cfg.get("app", "ml_sidecar_app")

    print("[AUTO] Writing cluster profiles…")
    write_kvstore_collection(
        cluster_profiles,
        base_url=base_url,
        token=token,
        app=app,
        collection="auth_cluster_profiles",
    )

    print("[AUTO] Writing user profiles…")
    write_kvstore_collection(
        user_profiles,
        base_url=base_url,
        token=token,
        app=app,
        collection="auth_user_profiles",
    )

    print("[AUTO] Writing event profiles…")
    write_kvstore_collection(
        event_records,
        base_url=base_url,
        token=token,
        app=app,
        collection="auth_events",
    )

    # ---- Günlük threshold koleksiyonu ----
    today = datetime.utcnow().date().isoformat()
    threshold_rec = [
        {
            "_key": today,
            "date": today,
            "threshold": float(threshold),
            "percentile": float(outlier_percentile),
            "event_count": int(len(scores)),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    ]

    print("[AUTO] Writing daily threshold…")
    write_kvstore_collection(
        threshold_rec,
        base_url=base_url,
        token=token,
        app=app,
        collection="auth_thresholds",
    )

    print("=== Auto-Pipeline Completed ===")
    return enriched


if __name__ == "__main__":
    run_auto_pipeline()