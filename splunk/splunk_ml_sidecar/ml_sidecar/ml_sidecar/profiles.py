from typing import List, Dict
from collections import defaultdict
import re
import ipaddress
import numpy as np

from ml_sidecar.utils_time import parse_windows_time


# -------------------------------------------
# Helper: private IP kontrolü
# -------------------------------------------
def _is_private_ip(ip: str) -> bool:
    if not ip:
        return False
    try:
        return ipaddress.ip_address(ip).is_private
    except Exception:
        return False


# -------------------------------------------
# Helper: KVStore _key üretimi (event)
# -------------------------------------------
def _make_event_key(evt: Dict) -> str:
    user = evt.get("user", "unknown")
    ts = evt.get("TimeCreated", "no_time")
    cid = evt.get("cluster_id", "na")

    key = f"{user}_{ts}_{cid}"
    key = key.replace(" ", "_")
    key = re.sub(r"[^A-Za-z0-9_\-:.]", "_", key)
    return key


# -------------------------------------------
# CLUSTER PROFILES
# -------------------------------------------
def build_cluster_profiles(events: List[Dict]) -> List[Dict]:
    """
    cluster_id bazında özet:
      - event_count
      - user_count
      - private_ip_rate
      - signature_distribution (dict)
      - label
    """
    if not events:
        return []

    cluster_users: Dict[int, set] = defaultdict(set)
    cluster_events: Dict[int, int] = defaultdict(int)
    cluster_private: Dict[int, int] = defaultdict(int)
    cluster_signatures: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for e in events:
        cid = int(e.get("cluster_id", -1))
        if cid < 0:
            continue

        user = e.get("user", "unknown")
        src = e.get("src")
        sig = e.get("signature") or "UNKNOWN"

        cluster_events[cid] += 1
        cluster_users[cid].add(user)
        cluster_signatures[cid][sig] += 1

        if _is_private_ip(src):
            cluster_private[cid] += 1

    profiles: List[Dict] = []

    for cid in sorted(cluster_events.keys()):
        ev_count = cluster_events[cid]
        user_count = len(cluster_users[cid]) or 1
        priv_rate = (cluster_private[cid] / ev_count) if ev_count else 0.0

        sig_counts = cluster_signatures[cid]
        total_sig = float(sum(sig_counts.values())) or 1.0
        sig_dist = {k: v / total_sig for k, v in sig_counts.items()}

        profiles.append(
            {
                "_key": f"cluster_{cid}",
                "cluster_id": int(cid),
                "event_count": int(ev_count),
                "user_count": int(user_count),
                "private_ip_rate": float(priv_rate),
                "signature_distribution": sig_dist,  # JSON obje olarak saklanacak
                "label": f"Cluster {cid}",
            }
        )

    return profiles


# -------------------------------------------
# USER PROFILES
# -------------------------------------------
def build_user_profiles(events: List[Dict]) -> List[Dict]:
    """
    user bazında:
      - dominant_cluster
      - confidence
      - mean_hour, std_hour
    """
    if not events:
        return []

    # cluster freq
    user_cluster_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    user_hours: Dict[str, List[int]] = defaultdict(list)

    for e in events:
        user = e.get("user", "unknown")
        cid = int(e.get("cluster_id", -1))
        if cid >= 0:
            user_cluster_counts[user][cid] += 1

        ts = e.get("TimeCreated")
        dt = parse_windows_time(ts)
        if dt:
            user_hours[user].append(dt.hour)

    profiles: List[Dict] = []

    for user, c_counts in user_cluster_counts.items():
        total = sum(c_counts.values()) or 1
        dominant_cluster = max(c_counts.items(), key=lambda kv: kv[1])[0]
        confidence = c_counts[dominant_cluster] / total

        hours = user_hours.get(user, [])
        if hours:
            mean_h = float(np.mean(hours))
            std_val = float(np.std(hours)) if len(hours) > 1 else 1.0
            std_h = std_val if std_val > 0 else 1.0
        else:
            mean_h = 12.0
            std_h = 4.0

        profiles.append(
            {
                "_key": f"user_{user}",
                "user": user,
                "dominant_cluster": int(dominant_cluster),
                "confidence": float(confidence),
                "mean_hour": float(mean_h),
                "std_hour": float(std_h),
            }
        )

    return profiles


# -------------------------------------------
# EVENT RECORDS (KVStore için minimal alan)
# -------------------------------------------
def build_event_records(events: List[Dict]) -> List[Dict]:
    """
    KVStore auth_events koleksiyonu için minimal alan seti:

      _key
      TimeCreated
      user
      src
      dest
      src_user
      signature_id
      signature
      action
      cluster_id
      final_anomaly_score
      behavior_outlier
    """
    records: List[Dict] = []

    for e in events:
        rec = {
            "_key": _make_event_key(e),
            "TimeCreated": e.get("TimeCreated"),
            "user": e.get("user"),
            "src": e.get("src"),
            "dest": e.get("dest"),
            "src_user": e.get("src_user"),
            "signature_id": e.get("signature_id"),
            "signature": e.get("signature"),
            "action": e.get("action"),
            "cluster_id": int(e.get("cluster_id", -1)) if e.get("cluster_id") is not None else -1,
            "final_anomaly_score": float(e.get("final_anomaly_score", 0.0)),
            "behavior_outlier": int(e.get("behavior_outlier", 0)),
        }
        records.append(rec)

    return records