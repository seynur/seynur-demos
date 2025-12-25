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
- Multi-layer anomaly scoring
- Per-user & global thresholding
- Configurable outlier decision logic
- Drift detection & auto-retraining
- Export to multiple KVStore collections
- Designed for dashboards, alerts, and investigations

The ML pipeline never writes to indexes — only to KVStore.


```
                           ┌────────────────────────────────────────┐
                           │        1) SPLUNK INGESTION (SEARCH)    │
                           └────────────────────────────────────────┘
                                          │
                                          ▼
                         Windows Authentication Events
                     (4624, 4625, 4672, Kerberos, NTLM, ...)
                                          │
                                          ▼
                          ┌────────────────────────────────────┐
                          │     2) FEATURE EXTRACTION          │
                          └────────────────────────────────────┘
                                          │
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ Extracted & derived features:                                                 │
 │  • hour, day_of_week                                                          │
 │  • signature_id                                                               │
 │  • action (success / failure)                                                 │
 │  • is_private_ip                                                              │
 │  • src subnet / dest subnet                                                   │
 │  • per-user temporal statistics (mean/std)                                    │
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
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ Auto-K selection                                                              │
 │  TRY K ∈ {6, 8, 10, 12, 14}                                                   │
 │   • Train KMeans                                                              │
 │   • Compute silhouette score                                                  │
 │   • Reject collapsed clusters                                                 │
 │   • Select best K                                                             │
 └───────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │ 5) RAW OUTLIER SCORE (CENTROID DISTANCE)       │
                └────────────────────────────────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │ 6) CONTEXTUAL BEHAVIOR SCORING                          │
                     └─────────────────────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ Supporting behavioral signals                                                  │
 │                                                                                │
 │ 1. cluster_rarity                                                              │
 │    = 1 − freq(user, cluster) / total(user events)                              │
 │                                                                                │
 │ 2. signature_rarity                                                            │
 │    = 1 − P(signature_id | cluster)                                             │
 │                                                                                │
 │ 3. user_hour_score                                                             │
 │    = min(|hour − mean_hour| / std_hour , 1)                                    │
 └────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────────┐
                      │   7) FINAL ANOMALY SCORE (WEIGHTED)        │
                      └────────────────────────────────────────────┘
                                          │
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ final_anomaly_score =                                                         │
 │   0.4 * raw_outlier_score                                                     │
 │ + 0.3 * cluster_rarity                                                        │
 │ + 0.2 * signature_rarity                                                      │
 │ + 0.1 * user_hour_score                                                       │
 └───────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │ 8) PER-USER NORMALIZATION                      │
                └────────────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ For each user:                                                                 │
 │  • Compute mean & std of final_anomaly_score                                   │
 │  • Calculate per_user_zscore                                                   │
 │  • Apply sigmoid → per_user_norm_score ∈ [0,1]                                 │
 │  • Derive per-user threshold (percentile-based)                                │
 └────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │ 9) GLOBAL THRESHOLDING                         │
                └────────────────────────────────────────────────┘
                                          │
                Compute global threshold on final_anomaly_score
                         (percentile-based, daily)
                                          │
                                          ▼
                ┌────────────────────────────────────────────────┐
                │ 10) FINAL OUTLIER DECISION LOGIC               │
                └────────────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ Configurable via settings.yaml                                                 │
 │                                                                                │
 │ outlier_mode = user | global | combined                                        │
 │ combined_logic = and | or                                                      │
 │                                                                                │
 │ behavior_outlier                                                               │
 │  • user      → per-user only                                                   │
 │  • global    → global only                                                     │
 │  • combined  → user AND/OR global                                              │
 │                                                                                │
 │ outlier_decision_source = user | global | combined_and | combined_or           │
 └────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌────────────────────────────────────────────────┐
                  │ 11) DRIFT DETECTION                            │
                  └────────────────────────────────────────────────┘
                                          │
                              Compare cluster_dist(old vs new)
                                          │
                            Chi-Square p-value < threshold ?
                              YES → Retrain model
                              NO  → Keep model
                                          │
                                          ▼
          ┌────────────────────────────────────────────────────────────────────┐
          │ 12) EXPORT RESULTS → KVSTORE COLLECTIONS                           │
          └────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
 │ auth_events                  │ auth_user_profiles           │ auth_cluster_profiles        │
 │ event-level enriched data    │ per-user behavior models     │ cluster summaries            │
 ├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
 │ auth_user_thresholds         │ auth_outlier_events (opt.)   │                              │
 │ per-user + global thresholds │ only outlier events          │                              │
 └──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
                                          │
                                          ▼
                     ┌────────────────────────────────────────────┐
                     │       SPLUNK DASHBOARDS & ALERTS           │
                     └────────────────────────────────────────────┘

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
Splunk Search
      ↓
ML Sidecar (Python Engine)
      ↓
Behavioral Modeling & Anomaly Detection
      ↓
KVStore Collections
      ↓
Splunk Dashboards & Alerts
```

Pipeline works as a “sidecar”:
- Reads authentication data from Splunk via REST
- Performs all ML computation externally
- Writes only enriched results back to Splunk KVStore
- Never modifies indexed data


```
ml_sidecar/
│
├── config/
│   └── settings.yaml          # All runtime configuration
│
├── core/
│   ├── config_loader.py       # YAML loader & validation
│   ├── ingestion.py           # Splunk REST ingestion
│   ├── features.py            # Feature extraction & encoding
│   ├── model.py               # Training, inference, drift detection
│   ├── pipeline.py            # End-to-end orchestration
│   ├── profiles.py            # User / cluster / event profiles
│   ├── kvstore.py             # KVStore export logic
│   └── utils.py               # Shared helpers
│
├── models/
│   └── <trained model artifacts>
│       ├── model.pkl
│       ├── scaler.pkl
│       └── meta.json
│
├── run_auto.py                # Entry point
└── README.md                  # This file
```

Each module serves a clear responsibility:
- `ingestion.py` → Splunk REST export → Python dict events
- `features.py` → raw event → numeric feature vector
- `model.py` → train/load KMeans + scaler, scoring, drift detection, metadata
- `pipeline.py` → orchestration (ingest → train/load → drift → predict → thresholds → export)
- `profiles.py` → build dashboard-friendly cluster/user/event records
- `kvstore.py` → KVStore delete + batch_save writes

---

# 3. Configuration (`config/settings.yaml`)

All runtime behavior is driven by this file.

Key sections:
- `general` → model path/name
- `ingestion` → search query + time window + Splunk REST details
- `modeling` → algorithm, k_candidates, drift threshold, outlier decision logic
- `output.splunk_kvstore` → KVStore write target

Example:

```
# =============================================================================
# ML SIDECAR — MASTER CONFIGURATION FILE
# =============================================================================
# This file defines:
#   - Model directory / name
#   - Splunk ingestion settings
#   - Modeling parameters
#   - Output destinations (Splunk KVStore)
#
# NOTES:
#   • Do NOT check real Splunk tokens into Git. Use environment variables instead.
#   • Time ranges follow Splunk syntax (e.g., -24h, -30d, now)
#   • All paths are relative to the ml_sidecar working directory
# =============================================================================


# -----------------------------------------------------------------------------
# GENERAL MODEL SETTINGS
# -----------------------------------------------------------------------------
general:
  # Directory where model .pkl and metadata .json will be stored
  model_dir: "./models"

  # Base name for KMeans model files
  model_name: "auth_kmeans_v1"

# -----------------------------------------------------------------------------
# INGESTION SETTINGS (SPLUNK REST API)
# -----------------------------------------------------------------------------
ingestion:
  # Query for fetching authentication logs
  query: 'index = ml_sidecar sourcetype = ml:sidecar:json'


  # Time window for Splunk data ingestion
  # Example -> earliest: "-24h"
  earliest: "-90d"
  latest: "now"

  # Splunk REST API access details
  splunk:
    # Example : "https://127.0.0.1:8089"
    base_url: "<base-url>"

    # IMPORTANT:
    #   For production use:
    #       auth_token: ${SPLUNK_TOKEN}
    #   Instead of storing the JWT directly inside the YAML.
    auth_token: <splunk bearer token for inputs>



# -----------------------------------------------------------------------------
# MODELING SETTINGS
# -----------------------------------------------------------------------------
modeling:
  algorithm: "kmeans"

  # Candidate K values used for silhouette-based selection
  k_candidates: [6, 8, 10, 12, 14]

  # Ensure reproducibility
  random_state: 42

  # Global drift threshold for model retraining (p-value)
  drift_threshold: 0.05

  # Decide final outlier flag based on mode  
  outlier_mode: "combined"   # Options -> "user" | "global" | "combined"
  combined_logic: "and" # Options -> "and" | "or"

  # Decide exporting outlier events
  export_outlier_events: true

# -----------------------------------------------------------------------------
# OUTPUT SETTINGS
# -----------------------------------------------------------------------------
output:
  # Only exporting to KVStore in this configuration
  writers:
    - "splunk_kvstore"

  # Splunk KVStore export configuration
  splunk_kvstore:
    enabled: true

    # Example -> "https://127.0.0.1:8089"
    base_url: "<base-url>"

    # Again — recommend using environment variables in production. 
    # Note : This Splunk can be different from ingestion.splunk.auth_token
    auth_token: <splunk bearer token for inputs>

    # Splunk App Name that contains the KVStore collections
    app: "ml_sidecar_app"

    # The specific collection to write
    collection: "auth_clusters"
```

---

# 4. Ingestion (core/ingestion.py)

The ingestion module uses:
- Splunk REST API
- /services/search/jobs/export
- output_mode=json
- Streaming results (line-by-line)

The output is a list of dict events, each containing fields such as:
- TimeCreated
- user
- src, dest
- signature_id
- action
- process
- raw authentication metadata

---

# 5. Features (core/features.py)

For each event, a numerical feature vector is produced.

Feature engineering combines:
- Extracted values from raw logs
- Computed behavioral indicators
- Learned priors from historical behavior

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
| Decision                     | behavior_outlier_user          | computed  | User percentile-based decision.                                     |
| Decision                     | behavior_outlier_cluster       | computed  | Clusterpercentile-based decision.                                   |
| Decision                     | behavior_outlier_global        | computed  | Global percentile-based decision.                                   |
| Decision                     | behavior_outlier               | computed  | Final combined outlier decision.                                    |
|------------------------------|--------------------------------|-----------|---------------------------------------------------------------------|





```

---

# 6. User Behavior Profiles (model.py)

Before training, the system builds a `user_profile`:

```
user → { mean_hour, std_hour }
```

This is used in:
- feature extraction
- temporal anomaly scoring
- per-user normalization

Profiles are stored inside the model metadata.

---

# 7. Model Training (core/model.py)

Training consists of:

Training follows a deterministic, multi-step process:
1. Build user behavior profiles
2. Extract numerical feature matrix
3. Scale features using MinMaxScaler
4. Perform Auto-K KMeans selection
5. Compute historical cluster distribution
6. Persist model artifacts:
    - model.pkl
    - scaler.pkl
    - meta.json

## Auto-K Selection

K ∈ `{6, 8, 10, 12, 14}`  
For each:

- Fit KMeans
- Compute silhouette score
- Reject collapsed solutions (all points → one cluster)
- Select best K

If all candidates fail, a safe fallback K (default: 4) is used to ensure pipeline continuity.

---

# 8. Prediction & Anomaly Scoring

Each event receives four anomaly components:
1. outlier_score: centroid distance
2. cluster_rarity:  user behavior deviation
3. signature_rarity: cluster signature mismatch
4. user_hour_score: temporal anomaly

### 8.1 `outlier_score`
This is the base anomaly signal produced directly by the clustering model.

- Defined as the distance to the assigned KMeans centroid
- Normalized to [0, 1]
- Captures structural deviation in feature space

### 8.2 `cluster_rarity`
This score measures how unusual the assigned cluster is for a specific user.

```
1 - freq(user, cluster) / total_user_events
```

Interpretation:
- Near 0 → user frequently appears in this cluster
- Near 1 → cluster is rare or unseen for that user

### 8.3. `signature_rarity`
This score measures how well the event’s signature fits its cluster.

```
1 - P(signature | cluster)
```

Where:
- P(signature | cluster) is learned from historical data
- Each cluster maintains its own signature distribution

Interpretation:
- Low value → signature is common in this cluster
- High value → signature is unexpected for the cluster

### 8.4. `user_hour_score`
This score measures how well the event’s signature fits its cluster.

Steps:
1.	Build per-user temporal profile:

    ```
    user → { mean_hour, std_hour }
    ```    

2.  Compute normalized deviation:  

    ```   
    min( |hour - mean_hour| / std_hour , 1 )
    ```

Interpretation:
- Near 0 → event occurred at a typical hour
- Near 1 → event occurred at an unusual time

### 8.5. Final Anomaly Score (Composite)
The final anomaly score is a weighted combination of all four signals:

```
0.4 * outlier +
0.3 * cluster_rarity +
0.2 * signature_rarity +
0.1 * user_hour_score
```

Design rationale:
- Structural deviation is dominant
- User behavior deviation is strongly weighted
- Contextual and temporal signals refine the decision

The result is a continuous score in [0, 1], not a hard verdict.

### 8.6. Adaptive Thresholding (User / Cluster / Global)
Instead of a fixed threshold (e.g. 0.7), the Sidecar uses distribution-aware thresholds. Thresholds are computed using percentiles:

`User Threshold`: 
Each user has their own behavioral baseline. Instead of comparing raw anomaly scores directly, the pipeline:
1.	Normalizes scores per user using z-score + sigmoid
2.	Computes a user-specific threshold on the normalized scores

`Cluster Threshold`: Captures cluster-specific variance (e.g. noisy VPN clusters vs stable service-account clusters). 

`Global Threshold`: The global threshold represents rare behavior across the entire environment (System-Wide Rarity).

`Combined Threshold`: Each event may be flagged by one or more threshold types:
- behavior_outlier_user
- behavior_outlier_cluster
- behavior_outlier_global

The final decision is configurable via settings.yaml:
```
modeling:
  # Decide final outlier flag based on mode  
  outlier_mode: "combined"   # Options -> "user" | "cluster" | "global" | "combined"

  # Threshold sources used when outlier_mode = combined
  combined_logic: "and" # Options -> "and" | "or"
  thresholds:
    enable_user: true
    enable_cluster: true
    enable_global: false
```

---

# 9. Drift Detection (core/model.py)

User authentication behavior is not static. Seasonal changes, operational shifts, migrations, or security incidents can all cause distributional drift in authentication patterns.

To keep the model aligned with current behavior, the Sidecar performs automatic drift detection using a Chi-Square test on cluster assignments.

Drift Evaluation Logic
1.	The trained model stores a historical cluster distribution:
```
old_dist = meta["cluster_dist"]
```

2.	New incoming events are projected onto the existing model:
```
new_dist = distribution(new_cluster_labels)
```

3.	A Chi-Square test is applied:
```
p = chisquare(new_dist, expected=old_dist)
```

Decision Rule
- p < drift_threshold → Drift detected
- Model is retrained automatically using current data
- p ≥ drift_threshold → No drift
- Existing model is reused

> Default threshold: 0.05. This value is configurable via modeling.drift_threshold in `settings.yaml`.

This approach ensures the model adapts to long-term behavioral changes without overreacting to short-term noise.

---

# 10. Profile Builders (core/profiles.py)

The pipeline produces 3 datasets:

### 10.1 Cluster Profiles (`auth_cluster_profiles`)
- cluster_id
- event_count
- user_count
- private_ip_rate
- signature_distribution
- label

### 10.2. User Profiles (`auth_user_profiles`)
Models long-term user behavior:
- user
- dominant_cluster — most frequent behavioral cluster
- confidence — strength of association with dominant cluster
- mean_hour — average authentication hour
- std_hour — temporal variability

User profiles are used both for:
- temporal anomaly scoring
- per-user adaptive thresholding

### 10.3. Event Records (`auth_events`)
Compact, enriched event-level records used directly by dashboards:
- Event metadata (TimeCreated, user, src, dest, signature)
- cluster_id
- anomaly components:
- outlier_score
- cluster_rarity
- signature_rarity
- user_hour_score
- final_anomaly_score
- outlier decisions:
  - behavior_outlier
  - behavior_outlier_user
  - behavior_outlier_cluster
  - behavior_outlier_global

Each record represents one authentication event with full behavioral context.

### 10.4. Outlier Event Records (`auth_outlier_events`)
Optional collection containing only events flagged as outliers. Export is controlled via:
```
modeling:
  export_outlier_events: true
```
Each record represents one authentication event with full behavioral context.

---

# 11. Export to Splunk KVStore (core/kvstore.py)
All model outputs are written to Splunk KVStore, not indexes.

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
1.	Load configuration (`settings.yaml`)
2.	Ingest authentication events from Splunk
3.	Train or load model artifacts
4.	Perform drift detection
5.	Run inference and anomaly scoring
6.	Apply adaptive thresholds:
    - user-based
    - cluster-based
    - global
7.	Resolve final outlier decision (user / cluster / global / combined)
8.	Build behavioral profiles
9.	Export results to KVStore

Thresholds are distribution-aware, not static:
```
threshold = quantile(scores, outlier_percentile)
```

Final decision logic is fully configurable via:
- outlier_mode
- combined_logic
- enabled threshold sources (user / cluster / global)

---

# 13. Running the Pipeline (`run_auto.py`)

```
python run_auto.py
```

This triggers:
- ingestion
- training or loading
- drift detection
- scoring
- thresholding
- profile generation
- KVStore export

Designed to run via:
- cron
- Splunk modular input
- external scheduler

---

# 14. Extensibility & Design Notes
The ML Sidecar is designed with production SOC environments in mind:

Key Design Principles
- Modular architecture
- Clear separation of concerns
- Streaming ingestion
- Deterministic scoring
- Config-driven behavior
- Multi-threshold decision logic
- No dependency on indexed ML results

Planned / Possible Extensions
- Additional ML algorithms (Isolation Forest, HDBSCAN, GMM)
- Incremental or online learning
- Per-sourcetype feature sets
- Sliding window inference
- Risk scoring aggregation
- Cross-user peer group modeling

---

# 15. Summary

The ML Sidecar is a complete behavioral analytics engine for Splunk:
- Feature engineering
- Adaptive clustering
- Multi-layer anomaly scoring
- Drift detection
- Flexible thresholding (user / cluster / global)
- KVStore-native output
- Dashboard-ready schemas

It enables behavior-first security analytics without modifying indexed data, while remaining transparent, explainable, and SOC-friendly.