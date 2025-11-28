#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — General utility helpers for the ML Sidecar

This module consolidates:
  - Filesystem utilities (directory creation)
  - Timestamp parsing utilities (Windows/Powershell formats, Splunk ISO timestamps)

Design goals:
  • Keep this module dependency-free (pure Python only)
  • Provide small, self-contained helpers for other modules
  • Avoid business logic (no model / pipeline / Splunk specifics here)

These utilities are used throughout ingestion, pipeline steps, profiles, and export.
"""

import os
from datetime import datetime


# ----------------------------------------------------------------------
# Filesystem Utilities
# ----------------------------------------------------------------------
def ensure_dir(path: str):
    """
    Ensure that a directory exists.
    
    Parameters
    ----------
    path : str
        Filesystem path to create if missing.

    Notes
    -----
    - Uses `exist_ok=True` so concurrent calls do not raise errors.
    - No-op if path is empty or None.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
# Timestamp Utilities
# ----------------------------------------------------------------------
def parse_windows_time(ts: str):
    """
    Parse Windows / Splunk ISO-like timestamps into a Python datetime object.

    Examples of supported formats:
        "2025-11-18T08:05:19.844741Z"
        "2024-02-10T12:19:33.1234567Z"   → trims extra digits
        "2025-05-18T18:20:45Z"           → no microseconds
        "2025-05-18T18:20:45.100Z"

    What this function fixes:
    -------------------------
    1. Converts trailing 'Z' → '+00:00' to fit Python's `fromisoformat()`.
    2. Python only accepts microseconds up to 6 digits, so extra digits are trimmed.
    3. If the timestamp is malformed or empty, returns None instead of raising.

    Parameters
    ----------
    ts : str
        The timestamp string to parse.

    Returns
    -------
    datetime or None
        Parsed timestamp, or None if parsing fails.
    """
    if not ts:
        return None

    try:
        # Convert trailing Z → timezone offset
        ts = ts.replace("Z", "+00:00")

        # Normalize microsecond precision:
        # Python enforces max 6 digits, but Windows sometimes produces 7+ digits.
        if "." in ts:
            date, rest = ts.split(".")
            micros, zone = rest.split("+", 1)
            micros = micros[:6]  # Trim excessive microseconds
            ts = f"{date}.{micros}+{zone}"

        return datetime.fromisoformat(ts)

    except Exception:
        # Graceful fallback
        return None