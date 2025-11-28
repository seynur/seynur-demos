#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profiles.py — Behavioral Profiles for Users, Clusters, and Events

This module builds the three profile structures that power the ML Sidecar’s
KVStore exports:

1) Cluster Profiles (auth_cluster_profiles)
   - Summaries of cluster-level behavior
   - Signature distributions, private IP rates, event counts, etc.

2) User Profiles (auth_user_profiles)
   - Per-user dominant cluster, behavior confidence, and temporal statistics

3) Event Records (auth_events)
   - Minimal enriched event representation suitable for dashboards

Design Notes
------------
• Profiles are intentionally compact — suitable for KVStore usage in Splunk.
• No model logic is present; these functions operate only on enriched events.
• Private IP logic and timestamp parsing are handled via small helpers.
"""

from typing import List, Dict
from collections import defaultdict
import re
import ipaddress
import numpy as np

from core.utils import parse_windows_time


# ============================================================================
# Internal Helpers
# ============================================================================

def _is_private_ip(ip: str) -> bool:
    """
    Return True if the given IP is RFC1918 private.
    """
    if not ip:
        return False
    try:
        return ipaddress.ip_address(ip).is_private
    except Exception:
        return False


def _make_event_key(evt: Dict) -> str:
    """
    Construct a stable KVStore _key for an enriched event.

    Key format:
        <user>_<timestamp>_<cluster>

    Non-alphanumeric characters are sanitized to avoid KVStore rejection.
    """
    user = evt.get("user", "unknown")
    ts = evt.get("TimeCreated", "no_time")
    cid = evt.get("cluster_id", "na")

    key = f"{user}_{ts}_{cid}"
    key = key.replace(" ", "_")
    key = re.sub(r"[^A-Za-z0-9_\-:.]", "_", key)
    return key


# ============================================================================
# Cluster Profiles
# ============================================================================

def build_cluster_profiles(events: List[Dict]) -> List[Dict]:
    """
    Build behavioral summaries for each cluster_id.

    Produces one KVStore record per cluster:
        _key
        cluster_id
        event_count
        user_count
        private_ip_rate
        signature_distribution (dict)
        label

    Notes
    -----
    - signature_distribution is a normalized probability distribution.
    - private_ip_rate is the percentage of events originating from private IPs.
    """
    if not events:
        return []

    cluster_users = defaultdict(set)
    cluster_events = defaultdict(int)
    cluster_private = defaultdict(int)
    cluster_signatures = defaultdict(lambda: defaultdict(int))

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

    profiles = []

    for cid in sorted(cluster_events.keys()):
        ev_count = cluster_events[cid]
        user_count = len(cluster_users[cid]) or 1
        priv_rate = cluster_private[cid] / ev_count if ev_count > 0 else 0.0

        sig_counts = cluster_signatures[cid]
        total_sig = float(sum(sig_counts.values())) or 1.0
        sig_dist = {k: v / total_sig for k, v in sig_counts.items()}

        profiles.append(
            {
                "_key": f"cluster_{cid}",
                "cluster_id": cid,
                "event_count": ev_count,
                "user_count": user_count,
                "private_ip_rate": priv_rate,
                "signature_distribution": sig_dist,
                "label": f"Cluster {cid}",
            }
        )

    return profiles


# ============================================================================
# User Profiles
# ============================================================================

def build_user_profiles(events: List[Dict]) -> List[Dict]:
    """
    Build user-level behavioral profiles.

    Produced fields:
        _key
        user
        dominant_cluster
        confidence
        mean_hour
        std_hour

    Definitions
    -----------
    dominant_cluster:
        The cluster the user hits most often.

    confidence:
        share_of_events_in_dominant_cluster = count(dominant) / total(user events)

    mean_hour / std_hour:
        Temporal behavior modeling from TimeCreated.
        If insufficient timestamps exist, defaults are applied.
    """
    if not events:
        return []

    user_cluster_counts = defaultdict(lambda: defaultdict(int))
    user_hours = defaultdict(list)

    for e in events:
        user = e.get("user", "unknown")
        cid = int(e.get("cluster_id", -1))
        if cid >= 0:
            user_cluster_counts[user][cid] += 1

        # User temporal behavior
        ts = e.get("TimeCreated")
        dt = parse_windows_time(ts)
        if dt:
            user_hours[user].append(dt.hour)

    profiles = []

    for user, c_counts in user_cluster_counts.items():
        total = sum(c_counts.values()) or 1

        dominant_cluster = max(c_counts.items(), key=lambda kv: kv[1])[0]
        confidence = c_counts[dominant_cluster] / total

        hours = user_hours.get(user, [])
        if hours:  # user has timestamp data
            mean_h = float(np.mean(hours))
            std_val = float(np.std(hours)) if len(hours) > 1 else 1.0
            std_h = std_val if std_val > 0 else 1.0
        else:  # fallback defaults
            mean_h = 12.0
            std_h = 4.0

        profiles.append(
            {
                "_key": f"user_{user}",
                "user": user,
                "dominant_cluster": dominant_cluster,
                "confidence": confidence,
                "mean_hour": mean_h,
                "std_hour": std_h,
            }
        )

    return profiles


# ============================================================================
# Event Records (Minimal KVStore Export)
# ============================================================================

def build_event_records(events: List[Dict]) -> List[Dict]:
    """
    Produce KVStore-ready, minimal enriched event documents.

    Useful for dashboards because:
        • Very small footprint per event
        • Stable _key
        • Contains only fields visualizations need

    Output fields:
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
    records = []

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
            "cluster_id": int(e.get("cluster_id", -1))
            if e.get("cluster_id") is not None
            else -1,
            "final_anomaly_score": float(e.get("final_anomaly_score", 0.0)),
            "behavior_outlier": int(e.get("behavior_outlier", 0)),
        }
        records.append(rec)

    return records