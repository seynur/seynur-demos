# 📌 1. Overview

ML Sidecar is a behavioral analytics engine designed to cluster Windows Authentication logs and compute multi-layer anomaly scores for every event.

It operates as a companion pipeline (“sidecar”) to Splunk:
Splunk → ML Sidecar (Python) → Enriched Events → KVStore → Dashboards.

Core Components:
- ML Pipeline (Python)
- Feature Extraction
- Adaptive KMeans Clustering (auto-K)
- 4-Layer Composite Anomaly Scoring
- Drift Detection & Auto-Retraining
- KVStore Export (3 collections)
- Splunk Dashboards (User, Cluster, Anomaly Explorer)

The system supports both batch (daily) and incremental operation.

---

# 📌 2. High-Level Pipeline Diagram

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

## 📌 2.1. Architecture Overview

```
Splunk Search → ML Sidecar (Python) → KMeans → Anomaly Scores
                                      ↓
                             KVStore Collections
                                      ↓
                           Splunk Dashboards & Alerts
```
Pipeline works as a “sidecar”:
Gets Splunk logs → processes → writes back to Splunk.

---

# 📌 3. Data Flow
## 3.1 Input

Input SPL (configurable):

```
index=wineventlog EventCode IN (4624,4625,4634,4672,4768,4769)
| table TimeCreated, user, src, dest, signature, signature_id, process, action, src_user
```

> As a default earliest time is 90 days. (This cdan be configured from the settings.yaml file.)

---

# 📌 4. Features
## 4.1 Feature Extraction

The below features are extracted for every authentication events and normalized with MinMaxScaler function.

```
Feature, Açıklama

hour, Etkinliğin saati
dow, Haftanın günü
is_private_ip, Kaynak IP özel ağ mı?
signature_id, 4624 / 4625 / 4672
src_octet, IP prefix normalizasyonu
hour_zscore(user), Kullanıcıya göre saat sapması

```

## 4.1 Model Fields
1. Time-based features

```
Feature → Descriptions

hour →  User peak login hour modeling (0–23). To understand user's temporal behaviour profile
day_of_week →  Weekly behavioral rhythms (0=Monday, ...). To understand seasonality of the behaviours.
user_hour_zscore →  (hour - mean_hour) / std_hour. Distance from user’s normal login hour.

```

2. Authentication behavior features

```
Feature/Açıklama

signature_id → Windows event ID (4624, 4625, 4672, 4768, 4769). Identity of the logon/failure event.
action_type → success / failure ([1/0]).
privileged_action_flag → 4672 (privileged logon) → 1.

```

3. Source/Destination network features
```
Feature/Açıklama

is_private_ip → if the ip is private (RFC1918), 1, else 0.
src_subnet → A.B normalize subnet (e.g.: 10.10).
dest_subnet → dest subnet (e.g.: 10.10).
external_ip_flag → 1 if source is public (“attacker-like” behaviour). ???

```

4. Statistical / distributional features
```
Feature/Açıklama

user_cluster_hist prior → Past cluster distribution per user.
user_mean_hour → Learned mean login hour.
user_std_hour → Variance in login timing.
```

---

# 📌 5. Clustering

KMeans runs with multiple candidate values:

```
K ∈ {6, 8, 10, 12, 14}
```

For each K:
- Train model
- Measure silhouette score
- Select best K

Output:
- cluster_id
- outlier_score (normalized centroid distance)

---

# 📌 6. Composite Anomaly Score (Final Score)

A four-layer hybrid anomaly score:

## 6.1 Layer 1 — Raw Outlier Score (outlier_score)

Centroid Distance in the cluster.

```
outlier_score = normalized(centroid_distance)
```

## 6.2 Layer 2 — Cluster Rarity (user-based)

The rate at which a user falls into the relevant cluster.

```
cluster_rarity = 1 - ( user_cluster_freq / user_total_events )

```

## 6.3 Layer 3 — Signature Rarity (cluster-level)

Based on the cluster's signature distribution:

```
signature_rarity = 1 - P(signature | cluster)
```

## 6.4 Layer 4 — User Hour Z-Score

Deviation from the user's own active hour profile.

```
user_hour_score = min( abs(hour - mean) / std , 1 )
```

## 📌 6.5 Final Anomaly Score Formula

```
final_anomaly_score =
    0.4 * outlier_score +
    0.3 * cluster_rarity +
    0.2 * signature_rarity +
    0.1 * user_hour_score
```

Binary Outlier Flag

```
behavior_outlier = 1 if final_anomaly_score >= 0.8 else 0

```

---

# 📌 7. Drift Detection

Cluster distributions are monitored for model stability.

1. Current model meta → cluster_dist
2. New dataset → new_labels
3. Chi-Square test:

```
p = chisquare(new_dist, expected=old_dist)
```
- p < threshold → drift detected (retrain)
- p ≥ threshold → model stable

Default threshold: 0.05

---

# 📌 8. Output: KVStore Collections

Pipeline fills 3 collections in the Splunk:

## 8.1 auth_events

All enriched events (most detailed lookup) - 1 row = 1 event. Contains all enriched events including anomaly scores.

Fields:
```
_key
TimeCreated
user
src
dest
src_user
signature
signature_id
action
process

cluster_id
outlier_score
cluster_rarity
signature_rarity
user_hour_score
final_anomaly_score
behavior_outlier
```

## 8.2 auth_user_profiles

User behaviour profiles:
Behavior model for each user.

```
user
dominant_cluster
mean_hour
std_hour
confidence
```

## 8.3 auth_cluster_profiles

Cluster behaviour profiles:
Summaries of each cluster (signature distribution, private IP rate, etc.)

```
cluster_id
event_count
user_count
private_ip_rate
signature_distribution.*
```

---

# 📌 9. Splunk Preparation (Before Running the ML Pipeline)
Before the ML Sidecar can write enriched results back into Splunk, three KVStore-backed lookups and a multi-panel dashboard must be created.
This section describes the full Splunk configuration used by the pipeline.

## 📌 9.1. KVStore Lookups (`transforms.conf`)

The ML pipeline writes three different data structures back into Splunk. Each structure is mapped to a KVStore collection via `transforms.conf`.

`transforms.conf`

```
[auth_cluster_profiles_lookup]
collection = auth_cluster_profiles
external_type = kvstore
fields_list = _key, cluster_id, event_count, user_count, hour_bin_mean, dow_mean, success_rate, private_ip_rate, signature_distribution, label

[auth_user_profiles_lookup]
collection = auth_user_profiles
external_type = kvstore
fields_list = _key, user, dominant_cluster, confidence, mean_hour, std_hour

[auth_events_lookup]
collection = auth_events
external_type = kvstore
fields_list = _key, TimeCreated, user, src, dest, src_user, signature_id, signature, action, cluster_id, final_anomaly_score, behavior_outlier
````

What this does:

- Creates three lookup definitions pointing to KVStore collections
- Allows | inputlookup auth_events_lookup to return the enriched event-level ML outputs
- Exposes user- and cluster-level behavioral profiles for dashboards and rules

## 📌 9.2. KVStore Collections (`collections.conf`)
The underlying KVStore schema is defined in `collections.conf`.

`collections.conf`

1️⃣ Cluster Profiles

```
[auth_cluster_profiles]
field.type._key = string
field.type.cluster_id = string
field.type.event_count = string
field.type.user_count = string
field.type.hour_bin_mean = string
field.type.dow_mean = string
field.type.success_rate = string
field.type.private_ip_rate = string
field.type.signature_distribution = string
field.type.label = string
```

2️⃣ User Profiles

```
[auth_user_profiles]
field.type._key = string
field.type.user = string
field.type.dominant_cluster = string
field.type.confidence = string
field.type.mean_hour = string
field.type.std_hour = string
```

3️⃣ Event-Level Enriched Records

```
[auth_events]
field.type.TimeCreated = string
field.type.user = string
field.type.src = string
field.type.dest = string
field.type.src_user = string
field.type.signature_id = number
field.type.signature = string
field.type.action = string
field.type.process = string
field.type.cluster_id = number
field.type.outlier_score = number
field.type.cluster_rarity = number
field.type.signature_rarity = number
field.type.user_hour_score = number
field.type.final_anomaly_score = number
field.type.behavior_outlier = number
```

## 📌 9.3. ML Dashboard (Authentication ML Anomaly Detection Test)

A full-featured Splunk Dashboard is provided, built with:
- 3 main panels (User Overview / Cluster Analytics / Anomaly Explorer)
- Dynamic dropdowns for:
- User
- Cluster ID
- Anomaly Score threshold
- Time-range picker
- 13 data sources (| inputlookup)
- Visualizations: Table, Line Chart, Pie, Bubble, Area, Entropy table


Dashboard Capabilities:

```
Panel → Description

User Behavior Overview →  Shows login patterns, cluster timeline, outliers over time
Cluster Analytics → Cluster characteristics, signature distribution, cluster-level entropy
Anomaly Explorer →Outliers sorted by score, bubble anomaly map, high-risk timeline
```

Key Metrics visualized:

- Cluster distribution per user
- Behavior outliers over time
- Signature frequency
- Final anomaly score timeline
- Shannon entropy per cluster/signature
- High-risk events (score > threshold)
- Dominant cluster ID per user

The full XML/JSON of the dashboard is included in the section you provided. In the README, we summarize it without embedding the full XML.

## 📌 9.4. How Splunk & ML Sidecar Interact

```
ML Sidecar → Updates KVStore collections
Splunk Dashboards → Read from KVStore via inputlookup
```

The ML pipeline never writes to indexes — only to KVStore, which is ideal for continuously updated behavioral profiles.

## 📌 9.5. Validation Commands

You can confirm correct configuration using:

```
| inputlookup auth_events_lookup | head 5
```

```
| inputlookup auth_user_profiles_lookup 
```

```
| inputlookup auth_cluster_profiles_lookup 
```

---

# 📌 10. Installation & Execution Steps

1. Clone the project

```
git clone https://github.com/seynur/seynur-demos
cd seynur-demos/ml_sidecar
```

2. Install the package in development mode
```
pip install -e .
```

This makes the package importable without reinstalling during development.


3. Configure `settings.yaml`
You must edit:

```
ml_sidecar/settings.yaml
```

Key fields to update:

```
ingestion:
  earliest: -90d
  latest: now
  search_query: <your splunk search>

general:
  model_dir: ./model/
  model_name: kmeans_model.pkl

modeling:
  train_ratio: 0.8
  drift_threshold: 0.05

output:
  file:
    path: ./output/events.json

  splunk_kvstore:
    enabled: true
    base_url: https://127.0.0.1:8089
    auth_token: <splunk bearer token>
```

4. Run the full ML pipeline

```
python run_auto.py
```

This will:
- Fetch events from Splunk
- Train or load the model
- Run drift detection
- Compute anomaly scores
- Build user + cluster + event profiles
- Write results to KVStore

5. Validate KVStore content
Example SPL:

```
| inputlookup auth_events_lookup | head 20
```