import os
import json

from .ingestion import load_splunk_events
from .model import (
    model_exists,
    train_model,
    load_model,
    predict,
    compute_model_drift,
)
from .utils import ensure_dir


def run_pipeline(config=None):
    """
    Full pipeline:
       - Fetch events from Splunk
       - Train model if not exists
       - Predict clusters for the same events
       - Write output to file
    """

    from ml_sidecar.config_loader import load_settings
    cfg = load_settings() if config is None else config

    # -------------------------------------------------------
    # 1. INGEST EVENTS
    # -------------------------------------------------------
    print("[PIPE] Fetching events...")

    events = load_splunk_events(cfg["ingestion"])
    print(f"[PIPE] Loaded {len(events)} events from Splunk")

    # -------------------------------------------------------
    # 2. TRAIN MODEL IF NOT EXISTS
    # -------------------------------------------------------
    model_path = os.path.join(cfg["general"]["model_dir"], cfg["general"]["model_name"])
    ensure_dir(cfg["general"]["model_dir"])

    if not model_exists(model_path):
        print("[PIPE] Model not found → training...")
        train_model(events, cfg, model_path)
    else:
        print("[PIPE] Existing model found → using saved model")

    # -------------------------------------------------------
    # 3. LOAD MODEL (model, scaler, meta)
    # -------------------------------------------------------
    model_tuple = load_model(model_path)

    # -------------------------------------------------------
    # 4. PREDICT
    # -------------------------------------------------------
    print("[PIPE] Running predictions...")
    enriched = predict(model_tuple, events)

    # -------------------------------------------------------
    # 5. WRITE OUTPUT
    # -------------------------------------------------------
    out_path = cfg["output"]["file"]["path"]
    ensure_dir(os.path.dirname(out_path))

    with open(out_path, "w") as f:
        for e in enriched:
            f.write(json.dumps(e) + "\n")

    print(f"[PIPE] Wrote results to {out_path}")

def run_inference_pipeline(config=None):
    from ml_sidecar.config_loader import load_settings

    cfg = load_settings() if config is None else config
    print("=== ML Sidecar Inference Run ===")
    print("[PIPE] Fetching events...")

    # 1. Fetch events
    events = load_splunk_events(cfg["ingestion"])
    print(f"[PIPE] Loaded {len(events)} events from Splunk")

    # 2. Load model (model, scaler, meta)
    model_path = os.path.join(cfg["general"]["model_dir"], cfg["general"]["model_name"])
    model, scaler, meta = load_model(model_path)
    print("[PIPE] Existing model found → using saved model")

    # 3. Predict
    print("[PIPE] Running predictions...")
    enriched = predict((model, scaler, meta), events)

    # 4. Write output file
    out_path = cfg["output"]["file"]["path"]
    ensure_dir(os.path.dirname(out_path))

    with open(out_path, "w") as f:
        for e in enriched:
            f.write(json.dumps(e) + "\n")

    print(f"[PIPE] Wrote results to {out_path}")

    # 5. Return enriched list so CLI script can show correct count
    print(f"=== Inference Completed: {len(enriched)} results ===")
    return enriched

import os
import json
from ml_sidecar.ingestion import load_splunk_events
from ml_sidecar.model import (
    model_exists,
    train_model,
    load_model,
    predict,
    compute_model_drift,
)
from ml_sidecar.utils import ensure_dir


def run_auto_pipeline(config=None):
    from ml_sidecar.config_loader import load_settings
    cfg = load_settings() if config is None else config

    print("\n=== ML Sidecar Auto-Pipeline ===")

    # --------------------------------------------------------------
    # 1. FETCH DATA FROM SPLUNK
    # --------------------------------------------------------------
    events = load_splunk_events(cfg["ingestion"])
    total = len(events)
    print(f"[AUTO] Loaded {total} events from Splunk")

    if total == 0:
        print("[AUTO] No events → Aborting")
        return []

    # --------------------------------------------------------------
    # 2. MODEL PATH
    # --------------------------------------------------------------
    model_path = os.path.join(cfg["general"]["model_dir"], cfg["general"]["model_name"])
    ensure_dir(cfg["general"]["model_dir"])

    # --------------------------------------------------------------
    # 3. NO MODEL: TRAIN FULL DATASET
    # --------------------------------------------------------------
    if not model_exists(model_path):
        print("[AUTO] No existing model → Training full dataset")
        model, scaler, meta = train_model(events, cfg, model_path)

        # After training, predict entire dataset
        enriched = predict((model, scaler, meta), events)

    else:
        # ----------------------------------------------------------
        # 4. MODEL EXISTS → LOAD + DRIFT CHECK
        # ----------------------------------------------------------
        print("[AUTO] Existing model found → Loading")
        model, scaler, meta = load_model(model_path)

        # Predict using existing model
        print("[AUTO] Predicting with existing model for drift detection")
        enriched_temp = predict((model, scaler, meta), events)
        new_labels = [e["cluster_id"] for e in enriched_temp]

        # Compute drift
        print("[AUTO] Computing model drift...")
        p_value = compute_model_drift(meta["cluster_dist"], new_labels)
        threshold = cfg["general"].get("drift_threshold", 0.05)

        print(f"[AUTO] Drift p-value = {p_value}")

        # ----------------------------------------------------------
        # 5. DRIFT DETECTED → RETRAIN
        # ----------------------------------------------------------
        if p_value > threshold:
            print("[AUTO] DRIFT DETECTED → Retraining model")

            split = int(total * 0.80)
            train_set = events[:split]
            test_set  = events[split:]

            print(f"[AUTO] Training on {len(train_set)} events")
            model, scaler, meta = train_model(train_set, cfg, model_path)

            print(f"[AUTO] Testing on {len(test_set)} events")
            _ = predict((model, scaler, meta), test_set)

            # Predict again after retrain
            enriched = predict((model, scaler, meta), events)

        else:
            print("[AUTO] Drift acceptable → Reusing existing model")
            enriched = enriched_temp

    # --------------------------------------------------------------
    # 6. WRITE RESULTS TO FILE
    # --------------------------------------------------------------
    out_path = cfg["output"]["file"]["path"]
    ensure_dir(os.path.dirname(out_path))

    with open(out_path, "w") as f:
        for e in enriched:
            f.write(json.dumps(e) + "\n")

    print(f"[AUTO] Wrote results to {out_path}")

    # --------------------------------------------------------------
    # 7. WRITE RESULTS TO SPLUNK KV STORE (IF ENABLED)
    # --------------------------------------------------------------
    kv_cfg = cfg["output"]["splunk_kvstore"]
    if kv_cfg.get("enabled", False):
        from ml_sidecar.kvstore import write_to_kvstore

        print("[AUTO] Writing enriched results to Splunk KV Store...")

        write_to_kvstore(
            enriched,
            base_url=kv_cfg["base_url"],
            auth_token=kv_cfg["auth_token"],
            app=kv_cfg["app"],
            collection=kv_cfg["collection"],
        )
        print("[AUTO] KV Store write completed.")
    else:
        print("[AUTO] KV Store disabled")

    print(f"[AUTO] Auto-Pipeline Complete — processed {len(enriched)} events\n")
    return enriched