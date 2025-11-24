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

from ml_sidecar.features import extract_features
from ml_sidecar.utils_time import parse_windows_time


# ---------------------------------------------------
# MODEL EXISTS?
# ---------------------------------------------------
def model_exists(path: str) -> bool:
    return os.path.exists(path + ".pkl")


# ---------------------------------------------------
# USER PROFILE (mean/std of login hour per user)
# ---------------------------------------------------
def build_user_profile(events: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Kullanıcı bazlı saat istatistiği:
    user -> { mean_hour, std_hour }
    """
    user_hours: Dict[str, List[int]] = {}

    for e in events:
        user = e.get("user", "unknown")
        ts = e.get("TimeCreated")
        if not ts:
            continue

        dt = parse_windows_time(ts)
        if not dt:
            continue

        user_hours.setdefault(user, []).append(dt.hour)

    profile: Dict[str, Dict[str, float]] = {}

    for user, hours in user_hours.items():
        if not hours:
            mean_h = 12.0
            std_h = 4.0
        else:
            mean_h = float(np.mean(hours))
            std_val = float(np.std(hours)) if len(hours) > 1 else 1.0
            std_h = std_val if std_val > 0 else 1.0

        profile[user] = {
            "mean_hour": mean_h,
            "std_hour": std_h,
        }

    return profile


# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
def train_model(events: List[Dict], config, save_path: str):
    """
    K-Means + MinMaxScaler + user_profile + cluster_dist üretir
    ve modeli disk'e kaydeder.
    """
    print(f"[MODEL] Training on {len(events)} events...")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1) user behavior profile
    user_profile = build_user_profile(events)

    # 2) feature matrisi
    X_raw = np.array(
        [extract_features(e, user_profile) for e in events],
        dtype="float32",
    )

    # 3) scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)

    # 4) candidate K seti
    ks = config.get("modeling", {}).get("candidate_k", [4, 6, 8, 10, 12])
    best_k = None
    best_model = None
    best_score = -1.0

    for k in ks:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X)

            # Tek cluster'a çökerse silhouette anlamsız
            if len(set(labels)) == 1:
                continue

            score = silhouette_score(X, labels)
            print(f"[MODEL] k={k} silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_model = km
                best_k = k
        except Exception as exc:
            print(f"[MODEL] k={k} failed: {exc}")
            continue

    if best_model is None:
        # Tamamen çökerse fallback
        print("[MODEL] Fallback k=4")
        best_model = KMeans(n_clusters=4, random_state=42, n_init="auto").fit(X)
        best_k = 4

    # 5) cluster dağılımı
    unique, counts = np.unique(best_model.labels_, return_counts=True)
    cluster_dist = {int(k): int(v) for k, v in zip(unique, counts)}

    # 6) user_profile'i JSON-safe hale getir
    safe_user_profile = {
        user: {
            "mean_hour": float(prof["mean_hour"]),
            "std_hour": float(prof["std_hour"]),
        }
        for user, prof in user_profile.items()
    }

    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "events": int(len(events)),
        "feature_dim": int(X.shape[1]),
        "best_k": int(best_k),
        "user_profile": safe_user_profile,
        "cluster_dist": cluster_dist,
    }

    def sanitize(o):
        if isinstance(o, dict):
            return {str(k): sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(v) for v in o]
        if hasattr(o, "item"):  # numpy scalar
            return o.item()
        return o

    meta = sanitize(meta)

    joblib.dump(best_model, save_path + ".pkl")
    joblib.dump(scaler, save_path + "_scaler.pkl")
    with open(save_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[MODEL] Saved model to {save_path}")
    return best_model, scaler, meta


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
def load_model(path: str):
    model = joblib.load(path + ".pkl")
    scaler = joblib.load(path + "_scaler.pkl")
    with open(path + ".json") as f:
        meta = json.load(f)
    return model, scaler, meta


# ---------------------------------------------------
# PREDICT + ANOMALY SCORE
# ---------------------------------------------------
def predict(model_data, events: List[Dict]) -> List[Dict]:
    """
    3 katmanlı anomaly skoru:

    - outlier_score: KMeans centroid distance, [0,1] normalize
    - cluster_rarity: user'ın bu cluster'a ne kadar az düştüğü (1 - freq)
    - signature_rarity: cluster içi signature olasılığına göre (1 - p(sig|cluster))
    - user_hour_score: user hour z-score, 3σ ve üstü ≈ 1
    - final_anomaly_score: 0.4 * outlier + 0.3 * cluster_rarity +
                           0.2 * signature_rarity + 0.1 * user_hour_score

    Bu fonksiyon behavior_outlier flag'ini set ETMEZ.
    Onu run_auto.py içinde threshold'a göre set ediyoruz.
    """
    model, scaler, meta = model_data
    user_profile = meta.get("user_profile", {})

    if not events:
        return []

    # ---------- 1. PASS: KMeans + histogramlar ----------
    X_raw = np.array(
        [extract_features(e, user_profile) for e in events],
        dtype="float32",
    )
    X_scaled = scaler.transform(X_raw)

    labels = model.predict(X_scaled)
    distances = model.transform(X_scaled)
    centroid_dist = distances.min(axis=1)

    # base outlier score [0,1]
    outlier_score = (centroid_dist - centroid_dist.min()) / (
        (centroid_dist.max() - centroid_dist.min()) + 1e-9
    )

    # user-cluster histogram: {user -> {cluster -> count}}
    user_cluster_hist: Dict[str, Dict[int, int]] = {}
    # cluster-signature histogram: {cluster -> {signature -> count}}
    cluster_sig_hist: Dict[int, Dict[str, int]] = {}

    for e, cid in zip(events, labels):
        cid = int(cid)
        user = e.get("user", "unknown")
        sig = e.get("signature")

        # user-cluster
        user_cluster_hist.setdefault(user, {})
        user_cluster_hist[user][cid] = user_cluster_hist[user].get(cid, 0) + 1

        # cluster-signature
        cluster_sig_hist.setdefault(cid, {})
        cluster_sig_hist[cid][sig] = cluster_sig_hist[cid].get(sig, 0) + 1

    total_user_events = {u: sum(d.values()) for u, d in user_cluster_hist.items()}

    # signature dağılımlarını olasılığa çevir
    cluster_sig_dist: Dict[int, Dict[str, float]] = {}
    for cid, sig_counts in cluster_sig_hist.items():
        total = float(sum(sig_counts.values())) or 1.0
        cluster_sig_dist[cid] = {sig: cnt / total for sig, cnt in sig_counts.items()}

    # ---------- 2. PASS: anomaly bileşenleri ----------
    output: List[Dict] = []

    for e, cid, base_out in zip(events, labels, outlier_score):
        cid = int(cid)
        user = e.get("user", "unknown")

        # cluster_rarity (user bazında)
        u_total = total_user_events.get(user, 1)
        u_c_freq = user_cluster_hist.get(user, {}).get(cid, 0)
        cluster_rarity = 1.0 - (u_c_freq / u_total)

        # signature_rarity (cluster içinde)
        sig = e.get("signature")
        sig_dist = cluster_sig_dist.get(cid, {})
        signature_rarity = 1.0 - sig_dist.get(sig, 0.0)

        # hour score
        ts = e.get("TimeCreated")
        dt = parse_windows_time(ts)
        if dt:
            hour = dt.hour
        else:
            hour = 12

        up = user_profile.get(user, {"mean_hour": 12.0, "std_hour": 4.0})
        mean_h = up.get("mean_hour", 12.0)
        std_h = up.get("std_hour", 4.0) or 1.0

        z = abs(hour - mean_h) / std_h
        user_hour_score = min(z / 3.0, 1.0)  # 3σ ve üzeri ~ 1

        final_score = (
            0.4 * float(base_out)
            + 0.3 * float(cluster_rarity)
            + 0.2 * float(signature_rarity)
            + 0.1 * float(user_hour_score)
        )

        out_evt = dict(e)
        out_evt.update(
            {
                "cluster_id": cid,
                "outlier_score": float(base_out),
                "cluster_rarity": float(cluster_rarity),
                "signature_rarity": float(signature_rarity),
                "user_hour_score": float(user_hour_score),
                "final_anomaly_score": float(final_score),
            }
        )
        output.append(out_evt)

    return output


# ---------------------------------------------------
# DRIFT
# ---------------------------------------------------
def compute_model_drift(old_dist, new_labels):
    """
    Eski cluster dağılımı ile yeni dağılımı karşılaştırır.
    p-value küçükse drift var kabul edilir.
    """
    old_dist_clean = {int(k): float(v) for k, v in old_dist.items()}

    uniq, cnt = np.unique(new_labels, return_counts=True)
    new_dist = {int(u): float(c) for u, c in zip(uniq, cnt)}

    keys = sorted(set(old_dist_clean.keys()) | set(new_dist.keys()))

    old_arr = np.array([old_dist_clean.get(k, 1e-6) for k in keys], dtype=float)
    new_arr = np.array([new_dist.get(k, 1e-6) for k in keys], dtype=float)

    old_sum = old_arr.sum()
    new_sum = new_arr.sum()

    if old_sum == 0 or new_sum == 0:
        return 1.0  # no drift anlamına gelsin

    old_arr = old_arr / old_sum
    new_arr = new_arr / new_sum

    try:
        chi2, p = chisquare(new_arr, f_exp=old_arr)
        return p
    except Exception as e:
        print("[DRIFT] WARNING:", e, "→ fallback p=1.0")
        return 1.0