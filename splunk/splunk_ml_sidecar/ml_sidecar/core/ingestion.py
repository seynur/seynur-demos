#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingestion.py — Splunk ingestion utilities.

This module provides a single function:
    load_splunk_events(conf)

Which:
    - Executes a Splunk REST search/jobs/export request  
    - Streams JSON events  
    - Parses both JSON-encoded _raw fields and structured SPL results  
    - Returns Python dictionaries for ML processing  

Notes:
    - This uses Splunk's streaming export API which does not create a job.
    - Authentication header varies by environment:
        * Cloud / JWT → Authorization: Bearer <token>
        * Classic Splunk token → Authorization: Splunk <token>
      The config determines which scheme is used.
"""

import requests
import json
import urllib3

urllib3.disable_warnings()   # Disable SSL warnings for local dev


def load_splunk_events(conf):
    """
    Load events from Splunk using the export API.

    Parameters
    ----------
    conf : dict
        Ingestion-related configuration, including:
            conf["query"]       → SPL query (without 'search')
            conf["earliest"]    → earliest time (default: -24h)
            conf["latest"]      → latest time (default: now)
            conf["splunk"]      → { base_url, auth_token }

    Returns
    -------
    events : list of dict
        Parsed event objects. Each entry is a Python dictionary produced from
        the event payload or _raw field when present.

    Behavior
    --------
    The function:
        1. Sends SPL query to /services/search/jobs/export
        2. Splits streamed JSON lines
        3. Decodes both:
                - _raw JSON strings
                - structured result fields
        4. Returns clean event dictionaries
    """
    # ----------------------------------------------------------------------
    # Extract Splunk REST config
    # ----------------------------------------------------------------------
    spl = conf["splunk"]
    base_url = spl["base_url"].rstrip("/")
    token = spl["auth_token"]

    query = conf["query"]
    earliest = conf.get("earliest", "-24h")
    latest = conf.get("latest", "now")

    # Splunk streaming export endpoint
    url = f"{base_url}/services/search/jobs/export"

    # Export API POST payload
    data = {
        "search": f"search {query}",
        "output_mode": "json",
        "earliest_time": earliest,
        "latest_time": latest,
    }

    # Authorization header — uses Bearer for JWT-style tokens
    # For classic Splunk tokens you'd use:
    #     "Authorization": f"Splunk {token}"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # ----------------------------------------------------------------------
    # Execute query
    # ----------------------------------------------------------------------
    print(f"[INGEST] Querying Splunk: {query}")

    try:
        resp = requests.post(url, data=data, headers=headers, verify=False)
    except Exception as e:
        print(f"[INGEST] ERROR: request failed → {e}")
        return []

    if resp.status_code not in (200, 201):
        print(f"[INGEST] ERROR: SPLUNK HTTP {resp.status_code}")
        print(resp.text[:500])
        return []

    # ----------------------------------------------------------------------
    # Parse streamed export lines
    # ----------------------------------------------------------------------
    lines = resp.text.splitlines()
    events = []

    print(f"[INGEST] Received {len(lines)} lines from export API.")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)   # Single JSON object from Splunk
        except json.JSONDecodeError:
            continue

        result = obj.get("result")
        if not result:
            continue

        # ------------------------------------------------------------------
        # CASE 1 — _raw contains JSON (best case)
        # ------------------------------------------------------------------
        raw_json = result.get("_raw")
        if raw_json:
            try:
                parsed = json.loads(raw_json)
                events.append(parsed)
                continue
            except Exception:
                pass

        # ------------------------------------------------------------------
        # CASE 2 — Parsed SPL fields (fallback)
        # ------------------------------------------------------------------
        events.append(result)

    print(f"[INGEST] Parsed {len(events)} events from Splunk.")
    return events