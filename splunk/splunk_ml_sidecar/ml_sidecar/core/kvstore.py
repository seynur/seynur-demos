#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kvstore.py — Splunk KVStore export utilities.

This module implements a high-performance batch writer for Splunk KVStore.
Typical use cases:
- Persist model output (events)  
- Store cluster profiles  
- Store user profiles  
- Store dynamic daily thresholds  

KVStore REST Endpoints used:
    DELETE  /storage/collections/data/<collection>
        → Clears the entire collection.

    POST    /storage/collections/data/<collection>/batch_save
        → Inserts multiple records at once.

Notes:
    - Authentication is via Splunk token:  Authorization: Splunk <token>
    - verify=False is used for local dev environments; in production this
      should be set to True or proper CA bundle should be provided.
"""

import json
import math
import requests


def write_kvstore_collection(
    records,
    base_url: str,
    token: str,
    app: str,
    collection: str,
    batch_size: int = 1000,
):
    """
    Completely replaces the contents of a KVStore collection with new records.

    Steps:
        1) DELETE collection (wipe existing data)
        2) POST in batches using Splunk's /batch_save endpoint

    Parameters
    ----------
    records : list of dict
        Records to write. Each record MUST include an "_key" field.

    base_url : str
        Splunk management API URL (e.g., https://127.0.0.1:8089)

    token : str
        Splunk authentication token (only the JWT value, not "Splunk <token>").

    app : str
        App context for KVStore (e.g., "ml_sidecar_app")

    collection : str
        KVStore collection name (e.g., "auth_events")

    batch_size : int
        Number of records per batch. Defaults to 1000.

    Returns
    -------
    None
        Writes output to KVStore and prints status messages.
    """
    if not records:
        print(f"[KVSTORE] No records to write for '{collection}'. Skipping.")
        return

    # ----------------------------------------------------------------------
    #  Build URLs and headers
    # ----------------------------------------------------------------------
    base = base_url.rstrip("/")
    headers = {
        "Authorization": f"Splunk {token}",
        "Content-Type": "application/json",
    }

    # ----------------------------------------------------------------------
    #  1. DELETE existing collection contents
    # ----------------------------------------------------------------------
    delete_url = (
        f"{base}/servicesNS/nobody/{app}/storage/collections/data/{collection}"
    )
    print(f"[KVSTORE] DELETE → {delete_url}")

    try:
        resp = requests.delete(delete_url, headers=headers, verify=False)
        print(f"[KVSTORE] DELETE status={resp.status_code}")
        if resp.status_code not in (200, 204):
            print(f"[KVSTORE] DELETE BODY: {resp.text}")
    except Exception as e:
        print(f"[KVSTORE] EXCEPTION during DELETE: {e}")

    # ----------------------------------------------------------------------
    #  2. POST records in batches using KVStore batch_save
    # ----------------------------------------------------------------------
    total = len(records)
    batches = int(math.ceil(total / float(batch_size)))

    for i in range(batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        chunk = records[start:end]

        post_url = (
            f"{base}/servicesNS/nobody/{app}/storage/collections/data/"
            f"{collection}/batch_save"
        )

        print(f"[KVSTORE] POST batch {i+1}/{batches} → {post_url} ({start}-{end})")

        try:
            resp = requests.post(
                post_url,
                data=json.dumps(chunk),
                headers=headers,
                verify=False,
            )

            if resp.status_code not in (200, 201):
                print(f"[KVSTORE] ERROR status={resp.status_code}")
                print(f"[KVSTORE] BODY: {resp.text}")
            else:
                print(f"[KVSTORE] OK status={resp.status_code}")

        except Exception as e:
            print(f"[KVSTORE] EXCEPTION during POST: {e}")

    print(f"[KVSTORE] Successfully wrote {total} records to '{collection}'.")