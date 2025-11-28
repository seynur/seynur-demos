# splunk_ml_sidecar — Repository Overview

This repository contains a full behavioral analytics pipeline (“ML Sidecar”) for Splunk, together with a Splunk app that provides dashboards and KVStore-backed lookups for visualizing the results.

The project consists of three major components:
1.	`ml_sidecar/` → Python ML engine (behavior modeling, clustering, anomaly scoring)
2.	`splunk_ml_sidecar_app/` → Splunk app (dashboards + KVStore collections)
3.	`auth-windows-log-generator-as-json-with-real-user-behaviour.py`→ Synthetic Windows authentication event generator

---

# 1. Directory Structure

```
splunk_ml_sidecar/
│
├── ml_sidecar/
│   ├── config/
|   |     └── settings.yaml
│   ├── etc/
|   |     ├──config_loader.py 
|   |     ├──features.py 
|   |     ├──ingestion.py
|   |     ├──kvstore.py 
|   |     ├──model.py 
|   |     ├──pipeline.py 
|   |     ├──profiles.py 
|   |     └──utils.py
│   ├── models
|   |     ├──...
|   |     └──<model-results>
│   ├── pyproject.toml
|   ├── run_auto.py
│   └── README.md   ← Detailed documentation for the ML engine
│
├── splunk_ml_sidecar_app/       ←  splunk app for dashboard and collections.
│   ├── default/
│   │     ├── transforms.conf         ← KVStore lookup definitions
│   │     ├── collections.conf         ← KVStore schemas
│   │     └── data/ui/views/…          ← Dashboard JSON (Dashboard Studio)
│   └── README.md
│
└── auth-windows-log-generator-as-json-with-real-user-behaviour.py
    ← Synthetic Windows authentication log generator
```

---

# 2. `ml_sidecar` (Python ML Engine)

The `ml_sidecar/` directory contains the entire machine learning engine responsible for:
- Fetching authentication events from Splunk via REST
- Extracting behavioral features
- Clustering via auto-K KMeans
- Computing multi-layer anomaly scores
- Detecting drift & retraining
- Writing results back to Splunk KVStore
- Producing three output datasets:
- auth_events
- auth_user_profiles
- auth_cluster_profiles

For detailed explanation of algorithms, features, scoring logic, drift detection, and output schema, see:

👉 `ml_sidecar/README.md`

---

# 3. `splunk_ml_sidecar_app` (Splunk Visualization Layer)
The splunk_ml_sidecar_app/ folder is a Splunk app containing:

KVStore collection configurations
- collections.conf
- transforms.conf

These allow the ML pipeline to write enriched results into Splunk lookups:
- auth_events_lookup
- auth_user_profiles_lookup
- auth_cluster_profiles_lookup

Dashboard Studio visualizations

Inside:
```
splunk_ml_sidecar_app/default/data/ui/views/
```
you will find full JSON dashboards for:
- User Behavior Explorer
- Cluster Analytics
- Anomaly Explorer
- Signature Distribution panel
- Outlier timeline
- Cluster entropy table
- Final score heatmaps

The dashboards are powered entirely by KVStore lookups updated by the ML sidecar.

See:
👉 `splunk_ml_sidecar_app/README.md`

---

# 4. Synthetic Windows Authentication Log Generator

This repository includes a realistic Windows authentication log generator:

`auth-windows-log-generator-as-json-with-real-user-behaviour.py`

It produces highly realistic behavioral patterns:
- uneven login frequencies across users
- variable src/dest IP distribution
- injected anomalies
- user-specific login hour patterns
- mix of success, failure, and privileged logons

Ideal for testing:
- ingestion
- feature extraction
- clustering
- drift detection
- dashboards

---

# 5. Quick Start

1. Install the ML Engine

```
cd ml_sidecar
pip install -e .
```

2. Configure Splunk REST token & query

Modify:
```
ml_sidecar/config/settings.yaml
```
Set:
- Splunk base URL
- REST token
- query
- earliest/latest time windows

3. Generate synthetic authentication logs

```
python3 auth-windows-log-generator-as-json-with-real-user-behaviour.py
```

4. Configure Splunk to ingest the output file

Update:
```
splunk_ml_sidecar_app/local/inputs.conf
```

Example: 
```
# splunk_ml_sidecar_app/local/inputs.conf
[monitor://<full-path-of-the-input-file>]
disabled = false
index = ml_sidecar
sourcetype = ml:sidecar:json
```

Restart Splunk after adding the app.

6. Validate KVStore content in Splunk
```
| inputlookup auth_events_lookup | head 20
```

```
| inputlookup auth_cluster_profiles_lookup
```

```
| inputlookup auth_user_profiles_lookup
```

---

# 6. Summary

This repository delivers a full behavioral analytics stack for Splunk:
- Python-based ML sidecar
- Real-time behavioral modeling
- Multi-layer anomaly scoring
- Automated drift detection & retraining
- KVStore integration
- Complete Splunk dashboard suite
- Synthetic data generator

The ML pipeline and Splunk app are cleanly decoupled but fully integrated—forming a robust, extensible architecture for advanced authentication analytics.