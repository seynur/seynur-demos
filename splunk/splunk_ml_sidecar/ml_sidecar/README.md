# ML Sidecar — Detailed Engine Documentation

This document provides the full internal technical description of the ML Sidecar engine under:

ml_sidecar/
    config/
    core/
    models/
    run_auto.py

It expands upon the high-level repository README and focuses on:
- The Python ML architecture
- Feature engineering
- Clustering & scoring internals
- Drift detection logic
- Pipeline orchestration
- Output schemas
- Design decisions and extensibility notes

---

# 1. Purpose and Architecture

ML Sidecar is designed as a companion behavioral analytics engine for Splunk.  
It reads Windows authentication logs, models user behavior, detects anomalies, and writes enriched records back into Splunk KVStore.

Core capabilities:
- REST ingestion from Splunk Search
- Feature extraction & normalization
- Adaptive KMeans clustering (auto-K)
- 4-layer anomaly scoring
- Drift detection & auto-retraining
- Exporting to multiple KVStore collections
- Support for dashboards, alerts, and investigations

The ML pipeline never writes to indexes — only to KVStore.


```
                           ┌────────────────────────────────────────┐
                           │        1) SPLUNK INGESTION (SEARCH)    │
                           └────────────────────────────────────────┘
                                          │
                                          ▼
                         "Authentication Events" (4624,4625,4672,..)
                                          │
                                          ▼
                          ┌────────────────────────────────────┐
                          │     2) FEATURE EXTRACTION          │
                          └────────────────────────────────────┘
                                          │
    ┌───────────────────────────────────────────────────────────────────────────────┐
    │ Extracted features:                                                           │
    │  • hour, day_of_week                                                          │
    │  • signature_id                                                               │
    │  • is_private_ip                                                              │
    │  • src subnet / dest subnet                                                   │
    │  • user_hour_zscore (hour-mean)/std                                           │
    └───────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                           ┌────────────────────────────────────────┐
                           │     3) SCALING (MinMaxScaler)          │
                           └────────────────────────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────────┐
                     │     4) KMEANS CLUSTERING (AUTO-K)           │
                     └─────────────────────────────────────────────┘
                                          │
                                          ▼
               ┌───────────────────────────────────────────────────────────────┐
               │ TRY K ∈ {6, 8, 10, 12, 14}                                     │
               │ For each K:                                                   │
               │    • Fit KMeans                                               │
               │    • Calculate silhouette score                               │
               │ Select best K                                                 │
               └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │ 5) RAW OUTLIER SCORE ( CENTROID DISTANCE)      │
                └────────────────────────────────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │ 6) BEHAVIOR ANALYSIS (3 SUPPORTING RARITY SCORES)       │
                     └─────────────────────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ SUPPORT SCORE #1 → cluster_rarity                                              │
 │    = 1 − freq(user, cluster) / total(user events)                              │
 │                                                                                │
 │ SUPPORT SCORE #2 → signature_rarity                                            │
 │    = 1 − P(signature | cluster)                                                │
 │                                                                                │
 │ SUPPORT SCORE #3 → user_hour_score (Z-score)                                   │
 │    = min(|hour - mean_hour| / std_hour , 1)                                    │
 └────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────────┐
                      │   7) FINAL ANOMALY SCORE (WEIGHTED)        │
                      └────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌───────────────────────────────────────────────────────────────────────────────┐
   │ final_anomaly_score =                                                         │
   │   0.4 * outlier_score                                                         │
   │ + 0.3 * cluster_rarity                                                        │
   │ + 0.2 * signature_rarity                                                      │
   │ + 0.1 * user_hour_score                                                       │
   │                                                                               │
   │ behavior_outlier = (final_score ≥ 0.8 ? 1 : 0)                                │
   └───────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌────────────────────────────────────────────────┐
                  │           8) DRIFT DETECTION                   │
                  └────────────────────────────────────────────────┘
                                          │
                              Compare cluster_dist(old vs new)
                                          │
                            Chi-Square p-value < threshold ?
                               YES → Retrain model
                               NO  → Keep model
                                          │
                                          ▼
       ┌───────────────────────────────────────────────────────────────────────────┐
       │              9) RESULT EXPORT → 3 KVSTORE COLLECTIONS                     │
       └───────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌───────────────────────────────────┬──────────────────────────────┬────────────────────────────┐
  │ auth_events                       │ auth_user_profiles           │ auth_cluster_profiles      │
  │ (event-level enriched records)    │ (user behavior model)        │ (cluster-level summaries)  │
  └───────────────────────────────────┴──────────────────────────────┴────────────────────────────┘
                                          │
                                          ▼
                     ┌────────────────────────────────────────────┐
                     │       10) SPLUNK DASHBOARDS                │
                     └────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌───────────────────────────────────────────────────────────────────────────────┐
   │  DASHBOARD VIEWS                                                              │
   │  - User Behavior Explorer                                                     │
   │  - Cluster Behavior Analyzer                                                  │
   │  - Anomaly Explorer (Top Outliers)                                            │
   │  - Entropy-based risk panel                                                   │
   │  - Final Score Trend Heatmap                                                  │
   │  - Signature Distribution Charts                                              │
   └───────────────────────────────────────────────────────────────────────────────┘

```

# 1.1. Input
Input SPL (configurable):

```
index=wineventlog EventCode IN (4624,4625,4634,4672,4768,4769)
| table TimeCreated, user, src, dest, signature, signature_id, process, action, src_user
```

> As a default earliest time is 90 days. (This cdan be configured from the settings.yaml file.)


---

# 2. Directory Structure (Engine Only)

```
Splunk Search → ML Sidecar (Python) → KMeans → Anomaly Scores
                                      ↓
                             KVStore Collections
                                      ↓
                           Splunk Dashboards & Alerts
```
Pipeline works as a “sidecar”:
Gets Splunk logs → processes → writes back to Splunk.

```
ml_sidecar/
│
├── config/
│   └── settings.yaml
│
├── core/
│   ├── config_loader.py
│   ├── ingestion.py
│   ├── features.py
│   ├── model.py
│   ├── pipeline.py
│   ├── profiles.py
│   ├── kvstore.py
│   └── utils.py
│
├── models/
│   └── <trained models .pkl/.json>
│
├── run_auto.py
└── README.md  ← this file
```

Each module serves a clear responsibility:
- ingestion.py → Splunk → Python
- features.py → Convert raw event → ML feature vector
- model.py → training, loading, scoring, drift detection
- profiles.py → build cluster/user/event profiles
- kvstore.py → exporting enriched output to Splunk
- pipeline.py → main orchestration logic

---

# 3. Configuration (`config/settings.yaml`)

All runtime behavior is driven by this file.

Example:

```
general:
  model_dir: "./models"
  model_name: "auth_kmeans_v1"
  drift_threshold: 0.05

ingestion:
  source_type: "splunk_rest"
  query: 'index=generic_index source="*20251120-win-synth-auth-as-json2.log"'
  earliest: "-365d"
  latest: "now"
  splunk:
    base_url: "<base-url>"
    auth_token: "<input-token>"

modeling:
  algorithm: "kmeans"
  k_candidates: [6, 8, 10, 12, 14]
  random_state: 42
  drift_threshold: 0.05

output:
  writers:
    - "splunk_kvstore"
  splunk_kvstore:
    enabled: true
    base_url: "<base-url>"
    auth_token: "<output-token>"
    app: "ml_sidecar_app"
    collection: "auth_clusters"
```

---

# 4. Ingestion (core/ingestion.py)

The ingestion module uses:
- Splunk REST API
- `/services/search/jobs/export`
- `output_mode=json`
- Streaming results line-by-line

It extracts:
- `_raw` JSON if present
- falls back to `result` dict

The output is a list of dict events, each containing fields such as:
- TimeCreated
- user
- src
- dest
- signature_id
- action
- src_user
- process

---

# 5. Features (core/features.py)

For each event, a numerical feature vector is produced.

Final feature set used in modeling:

```

|------------------------------|--------------------------------|-----------|---------------------------------------------------------------------|
| Category                     | Feature                        | Type      | Description                                                         |
|------------------------------|--------------------------------|-----------|---------------------------------------------------------------------|
| Time-based                   | hour                           | extracted | Event hour (0–23). Used for modelling temporal behavior.            |
| Time-based                   | day_of_week                    | extracted | Day of the week (0=Monday). Captures weekly patterns.               |
| Time-based                   | user_mean_hour                 | learned   | Average login hour learned from user profiles.                      |
| Time-based                   | user_std_hour                  | learned   | Login hour variability (standard deviation).                        |
| Time-based                   | user_hour_zscore               | computed  | (hour - mean_hour) / std_hour. Deviation from user's typical hour.  |
|------------------------------|--------------------------------|-----------|---------------------------------------------------------------------|
| Authentication behavior      | signature_id                   | extracted | Windows Event ID (4624,4625,4672,4768,4769).                        |
| Authentication behavior      | action_type                    | computed  | Encoded success/failure (1/0).                                      |
| Authentication behavior      | privileged_action_flag         | computed  | Flag for privileged logon (signature_id=4672).                      |
|------------------------------|--------------------------------|-----------|---------------------------------------------------------------------|
| Network                      | is_private_ip                  | computed  | 1 if RFC1918 private IP, else 0.                                    |
| Network                      | external_ip_flag               | computed  | 1 if external/public source IP.                                     |
| Network                      | src_subnet                     | computed  | Normalized source subnet (A.B format).                              |
| Network                      | dest_subnet                    | computed  | Normalized destination subnet (A.B format).                         |
|------------------------------|--------------------------------|-----------|---------------------------------------------------------------------|
| Statistical priors           | user_cluster_hist_prior        | learned   | Historical cluster distribution per user.                           |
| Statistical priors           | cluster_signature_distribution | learned   | Probability distribution of signatures in each cluster.             |
|------------------------------|--------------------------------|-----------|---------------------------------------------------------------------|
| Cluster outputs              | cluster_id                     | computed  | Assigned KMeans cluster ID.                                         |
| Cluster outputs              | outlier_score                  | computed  | Normalized centroid distance (raw anomaly score).                   |
|------------------------------|--------------------------------|-----------|---------------------------------------------------------------------|
| Anomaly layers               | cluster_rarity                 | computed  | 1 - (freq(user, cluster) / total user events).                      |
| Anomaly layers               | signature_rarity               | computed  | 1 - P(signature | cluster).                                         |
| Anomaly layers               | user_hour_score                | computed  | Time deviation from user's profile (0–1 clipped).                   |
| Anomaly layers               | final_anomaly_score            | computed  | Weighted anomaly score across 4 dimensions.                         |
| Anomaly layers               | behavior_outlier               | computed  | 1 if final_anomaly_score >= 0.8 else 0.                             |
|------------------------------|--------------------------------|-----------|---------------------------------------------------------------------|
```

---

# 6. User Behavior Profiles (model.py)

Before training, the system builds a `user_profile`:

```
user → { mean_hour, std_hour }
```

This is used in:
- feature extraction (hour_z)
- anomaly scoring (user_hour_score)

These profiles are saved alongside the model meta.

---

# 7. Model Training (core/model.py)

Training consists of:

1. Build user profile  
2. Extract feature matrix  
3. Scale with MinMaxScaler  
4. Auto-K selection  
5. Build cluster distribution  
6. Save:
   - model.pkl
   - scaler.pkl
   - meta.json

## Auto-K Selection

K ∈ `{6, 8, 10, 12, 14}`  
For each:

- Fit KMeans
- Compute silhouette score
- Select best K

If clustering collapses (all points → 1 cluster), the candidate is skipped.

---

# 8. Prediction & Scoring

Each event receives:

### 8.1 outlier_score
Normalized centroid distance.

### 8.2 cluster_rarity
```
1 - freq(user, cluster) / total_user_events
```

### 8.3 signature_rarity
```
1 - P(signature | cluster)
```

### 8.4 user_hour_score
Normalized Z-score:
```
min(|hour - mean| / std, 1)
```

### 8.5 Final Anomaly Score
```
0.4*outlier +
0.3*cluster_rarity +
0.2*signature_rarity +
0.1*user_hour_score
```

threshold:
```
behavior_outlier = 1 if final_anomaly_score ≥ 0.7
```

---

# 9. Drift Detection

Drift is evaluated using chi-square:

```
old_dist (from meta)
new_dist (from current dataset)
p = chisquare(new_dist, expected=old_dist)
```

Decision rule:
- p < threshold → drift detected → retrain
- else → keep existing model

> Default threshold: 0.05

---

# 10. Profile Builders (core/profiles.py)

The pipeline produces 3 datasets:

### 10.1 Cluster Profiles
- cluster_id
- event_count
- user_count
- private_ip_rate
- signature_distribution

### 10.2 User Profiles
- user
- dominant_cluster
- confidence
- mean_hour
- std_hour

### 10.3 Event Records
Minimal enriched records for dashboards:
- event metadata
- cluster_id
- anomaly scores
- `behavior_outlier`

---

# 11. Export to Splunk KVStore (core/kvstore.py)

The exporter supports:

### Delete entire collection
```
DELETE /storage/collections/data/<collection>
```

### Batch write (1000 items)
```
POST /storage/collections/data/<collection>/batch_save
```

---

# 12. Pipeline Orchestration (core/pipeline.py)

The main executor performs:

1. Load settings  
2. Ingest events  
3. Train or load model  
4. Drift detection  
5. Prediction  
6. Thresholding  
7. Build profiles  
8. Write to KVStore  

Threshold is dynamic:
```
threshold = quantile(scores, outlier_percentile)
```

---

# 13. Running the Pipeline (`run_auto.py`)

```
python run_auto.py
```

This runs the full chain:
- ingest
- train/load
- detect drift
- score
- export

---

# 14. Extensibility & Design Notes

Designed for production:
- modular architecture
- clean boundaries
- streaming ingestion
- multi-collection output
- deterministic scoring
- retraining logic built-in

Intended future improvements:
- support for additional ML algorithms
- incremental update mode
- batch inference without full export
- pluggable feature sets per sourcetype
- removing dependency on large JSON outputs

---

# 15. Summary

The ML Sidecar engine is a complete, modular behavioral analytics stack for Splunk:
- feature engineering
- clustering
- anomaly scoring
- drift detection
- KVStore export
- dashboard-ready schemas

It allows Splunk dashboards to operate on always-fresh behavioral insights without modifying any indexed data.
