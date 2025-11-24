# ml_sidecar/ingestion.py

import requests
import json
import urllib3
urllib3.disable_warnings()


def load_splunk_events(conf):
    # -------------------------------------------
    # Splunk config is under conf["splunk"]
    # -------------------------------------------
    spl = conf["splunk"]

    base_url = spl["base_url"]
    token = spl["auth_token"]

    query = conf["query"]
    earliest = conf.get("earliest", "-24h")
    latest = conf.get("latest", "now")

    url = f"{base_url}/services/search/jobs/export"

    data = {
        "search": f"search {query}",
        "output_mode": "json",
        "earliest_time": earliest,
        "latest_time": latest
    }

    headers = {
        "Authorization": f"Bearer {token}"
    }

    resp = requests.post(url, data=data, headers=headers, verify=False)

    lines = resp.text.splitlines()
    events = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except:
            continue

        result = obj.get("result")
        if not result:
            continue

        # ------------------------------
        # 1) Parse real event JSON from _raw
        # ------------------------------
        raw_json = result.get("_raw")
        if raw_json:
            try:
                parsed = json.loads(raw_json)
                events.append(parsed)
                continue
            except:
                pass

        # ------------------------------
        # 2) Accept parsed results also
        # ------------------------------
        events.append(result)

    return events