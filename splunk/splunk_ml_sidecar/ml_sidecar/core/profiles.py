#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profiles.py — Behavioral Profiles for Users, Clusters, and Events

This module constructs the three profile structures written to Splunk KVStore:

1) Cluster Profiles (auth_cluster_profiles)
   - Summaries of cluster-level behavior
   - Signature distributions, private IP rates, event counts

2) User Profiles (auth_user_profiles)
   - Per-user dominant cluster, confidence score, temporal statistics

3) Event Records (auth_events)
   - Compact enriched event document for dashboards

These profiles are intentionally lightweight and used by Splunk dashboards.
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
    Check whether an IP address is RFC1918 private.

    Returns True for:
        10.0.0.0/8
        172.16.0.0/12
        192.168.0.0/16

    Returns False for invalid or empty input.
    """
    if not ip:
        return False
    try:
        return ipaddress.ip_address(ip).is_private
    except Exception:
        # if parsing fails (corrupt IP), treat as non-private
        return False


def _make_event_key(evt: Dict) -> str:
    """
    Construct a stable KVStore `_key` for an enriched event.

    Format:
        <user>_<timestamp>_<cluster>

    • Keys must be deterministic.
    • Illegal KVStore characters are replaced with underscores.
    """
    user = evt.get("user", "unknown")
    ts = evt.get("TimeCreated", "no_time")
    cid = evt.get("cluster_id", "na")

    key = f"{user}_{ts}_{cid}"

    # sanitize: only allow alphanumeric, underscore, hyphen, colon, dot
    key = re.sub(r"[^A-Za-z0-9_\-:.]", "_", key)
    return key


# ============================================================================
# Cluster Profiles
# ============================================================================

def build_cluster_profiles(events: List[Dict]) -> List[Dict]:
    """
    Build aggregated behavioral summaries for each cluster_id.

    For each cluster, we compute:
        • total events
        • number of unique users
        • private IP rate
        • signature_id distribution (normalized probability)
        • label (for dashboard display)

    Returns:
        List[dict] — one record per cluster_id.
    """
    if not events:
        return []

    # Per-cluster tracking structures
    cluster_users = defaultdict(set)                   # unique users in cluster
    cluster_events = defaultdict(int)                  # event count
    cluster_private = defaultdict(int)                 # private src IP count

    # signature_id histogram per cluster
    cluster_signatures = defaultdict(lambda: defaultdict(int))

    # --- Aggregate counts for each cluster ---
    for e in events:
        cid = int(e.get("cluster_id", -1))
        if cid < 0:
            # skip events without valid cluster assignment
            continue

        user = e.get("user", "unknown")
        src = e.get("src")
        sig_id = str(e.get("signature_id") or "0")     # ensure string key

        cluster_events[cid] += 1
        cluster_users[cid].add(user)
        cluster_signatures[cid][sig_id] += 1

        if _is_private_ip(src):
            cluster_private[cid] += 1

    profiles = []

    # --- Build profile for each cluster ---
    for cid in sorted(cluster_events.keys()):
        ev_count = cluster_events[cid]
        user_count = len(cluster_users[cid]) or 1
        priv_rate = cluster_private[cid] / ev_count if ev_count > 0 else 0.0

        # Normalize signature distribution into probabilities
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
    Build per-user behavioral profiles for dashboards and scoring.

    Extracted per user:
        • dominant_cluster      → most frequently assigned cluster
        • confidence            → how strongly user belongs to dominant cluster
        • mean_hour, std_hour  → temporal login behavior model

    Returns:
        List[dict] — one record per user.
    """
    if not events:
        return []

    user_cluster_counts = defaultdict(lambda: defaultdict(int))
    user_hours = defaultdict(list)

    # --- Aggregate per-user data ---
    for e in events:
        user = e.get("user", "unknown")
        cid = int(e.get("cluster_id", -1))

        # count cluster assignment
        if cid >= 0:
            user_cluster_counts[user][cid] += 1

        # extract login hour for temporal modeling
        dt = parse_windows_time(e.get("TimeCreated"))
        if dt:
            user_hours[user].append(dt.hour)

    profiles = []

    # --- Build user profile documents ---
    for user, c_counts in user_cluster_counts.items():
        total = sum(c_counts.values()) or 1

        # dominant cluster (highest frequency)
        dominant_cluster = max(c_counts.items(), key=lambda x: x[1])[0]
        confidence = c_counts[dominant_cluster] / total

        # temporal statistics
        hours = user_hours.get(user, [])
        if hours:
            mean_h = float(np.mean(hours))
            std_val = float(np.std(hours)) if len(hours) > 1 else 1.0
            std_h = max(std_val, 1.0)  # avoid std = 0
        else:
            # fallback values for users with 0 timestamped events
            mean_h, std_h = 12.0, 4.0

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
# Event Records
# ============================================================================

def build_event_records(events: List[Dict]) -> List[Dict]:
    """
    Build minimal-enriched event records for KVStore.

    These entries are used directly by dashboards and should remain compact.

    Output fields include:
        • metadata fields (TimeCreated, user, src, dest, etc.)
        • model outputs (cluster_id, rarity scores, anomaly score)
        • behavior_outlier flag

    Returns:
        List[dict] — one record per event.
    """
    records = []

    for e in events:
        rec = {
            "_key": _make_event_key(e),                    # deterministic KV key
            "TimeCreated": e.get("TimeCreated"),
            "user": e.get("user"),
            "src": e.get("src"),
            "dest": e.get("dest"),
            "src_user": e.get("src_user"),

            # Windows authentication metadata
            "signature_id": e.get("signature_id"),
            "signature": e.get("signature"),               # used for dashboards
            "action": e.get("action"),

            # cluster assignment
            "cluster_id": int(e.get("cluster_id", -1)),

            # anomaly model outputs
            "outlier_score": float(e.get("outlier_score", 0.0)),
            "cluster_rarity": float(e.get("cluster_rarity", 0.0)),
            "signature_rarity": float(e.get("signature_rarity", 0.0)),
            "user_hour_score": float(e.get("user_hour_score", 0.0)),
            "final_anomaly_score": float(e.get("final_anomaly_score", 0.0)),

            # binary classification: above threshold?
            "behavior_outlier": int(e.get("behavior_outlier", 0)),
        }

        records.append(rec)

    return records