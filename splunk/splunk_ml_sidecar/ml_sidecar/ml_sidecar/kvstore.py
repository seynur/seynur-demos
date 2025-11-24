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
    KVStore koleksiyonunu tamamen temizleyip
    verilen kayıtları batch_save ile yükler.

    base_url:  https://127.0.0.1:8089
    token:     sadece JWT kısmı (Authorization: Splunk <token>)
    app:       ml_sidecar_app
    collection: auth_events / auth_user_profiles / auth_cluster_profiles / auth_thresholds
    """
    if not records:
        print(f"[KVSTORE] No records to write for '{collection}'")
        return

    base = base_url.rstrip("/")
    headers = {
        "Authorization": f"Splunk {token}",
        "Content-Type": "application/json",
    }

    # 1) koleksiyonu temizle
    del_url = f"{base}/servicesNS/nobody/{app}/storage/collections/data/{collection}"
    print(f"[KVSTORE] DELETE → {del_url}")
    try:
        resp = requests.delete(del_url, headers=headers, verify=False)
        print(f"[KVSTORE] DELETE status: {resp.status_code}")
        if resp.status_code not in (200, 204):
            print(f"[KVSTORE] DELETE body: {resp.text}")
    except Exception as e:
        print(f"[KVSTORE] DELETE error: {e}")

    # 2) batch_save
    total = len(records)
    batches = int(math.ceil(total / float(batch_size)))

    for i in range(batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        chunk = records[start:end]

        url = f"{base}/servicesNS/nobody/{app}/storage/collections/data/{collection}/batch_save"
        print(f"[KVSTORE] POST batch {i+1}/{batches} → {url} ({start}-{end})")

        try:
            resp = requests.post(
                url,
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
            print(f"[KVSTORE] POST error: {e}")

    print(f"[KVSTORE] Successfully wrote {total} records to '{collection}'")