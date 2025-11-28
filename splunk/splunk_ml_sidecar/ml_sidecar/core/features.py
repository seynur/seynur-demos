#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features.py — Feature engineering utilities for ML Sidecar.

This module defines:
    extract_features(evt, user_profile)

Which converts a raw Windows authentication event into a
numeric feature vector used by the clustering model.

Feature vector includes:
    - hour_bin: Normalized login hour (0–1)
    - success_flag: 1=success, 0=failure
    - signature_id: Numeric signature identifier
    - src_is_private: Source IP is private (1/0)
    - dest_is_private: Destination IP is private (1/0)
    - hour_zscore: Deviation from user's normal login hour (Z-score)

All features are returned as a NumPy float32 vector.
"""

import numpy as np
from core.utils import parse_windows_time


def extract_features(evt, user_profile):
    """
    Convert an authentication event into a numeric feature vector.

    Parameters
    ----------
    evt : dict
        Single event dictionary parsed from log ingestion.
    user_profile : dict
        Per-user statistics generated during model training:
            user_profile[user] = { "mean_hour": x, "std_hour": y }

    Returns
    -------
    np.ndarray (float32)
        Feature vector:
            [ hour_bin,
              success_flag,
              signature_id,
              src_is_private,
              dest_is_private,
              hour_z_score ]
    """

    # ----------------------------------------------------------------------
    # 1) Hour / Time features
    # ----------------------------------------------------------------------
    ts = evt.get("TimeCreated")
    dt = parse_windows_time(ts)

    if dt:
        hour = dt.hour
        hour_bin = hour / 23.0           # Normalized 0–1
    else:
        hour = 12
        hour_bin = 0.5

    # ----------------------------------------------------------------------
    # 2) Success / Failure flag
    # ----------------------------------------------------------------------
    success_flag = 1.0 if evt.get("action") == "success" else 0.0

    # ----------------------------------------------------------------------
    # 3) Signature ID (already numeric)
    # ----------------------------------------------------------------------
    sig = float(evt.get("signature_id", 0))

    # ----------------------------------------------------------------------
    # 4) Private IP detection
    # ----------------------------------------------------------------------
    def is_private(ip: str) -> bool:
        """
        Fast string-based private IP detection.
        Covers RFC1918 ranges:
            10.0.0.0/8
            172.16.0.0/12
            192.168.0.0/16
        """
        if not ip:
            return False
        return (
            ip.startswith("10.") or
            ip.startswith("192.168.") or
            ip.startswith("172.") and any(
                ip.startswith(f"172.{i}.") for i in range(16, 32)
            )
        )

    src = evt.get("src", "")
    dest = evt.get("dest", "")

    src_private = 1.0 if is_private(src) else 0.0
    dest_private = 1.0 if is_private(dest) else 0.0

    # ----------------------------------------------------------------------
    # 5) Deviation from user's typical login hour
    # ----------------------------------------------------------------------
    user = evt.get("user", "unknown")
    prof = user_profile.get(user, {"mean_hour": 12, "std_hour": 4})

    mean_h = prof.get("mean_hour", 12.0)
    std_h = prof.get("std_hour", 4.0) or 1.0  # Avoid zero division

    hour_dev = abs(hour - mean_h)
    hour_z = hour_dev / (std_h + 0.1)         # Stabilized Z-score

    # ----------------------------------------------------------------------
    # 6) Final numeric vector
    # ----------------------------------------------------------------------
    return np.array(
        [
            hour_bin,
            success_flag,
            sig,
            src_private,
            dest_private,
            hour_z,
        ],
        dtype="float32",
    )