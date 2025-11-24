import numpy as np
from ml_sidecar.utils_time import parse_windows_time

def extract_features(evt, user_profile):
    """
    Final feature set:
      - hour_bin
      - success_flag
      - signature_id (scaled by model)
      - src_is_private
      - dest_is_private
      - deviation_from_user_hour_profile
    """

    ts = evt.get("TimeCreated")
    dt = parse_windows_time(ts)

    if dt:
        hour = dt.hour
        hour_bin = hour / 23.0
    else:
        hour = 12
        hour_bin = 0.5

    # success/failure
    success_flag = 1.0 if evt.get("action") == "success" else 0.0

    # signature_id as numeric
    sig = float(evt.get("signature_id", 0))

    # src private?
    src = evt.get("src", "")
    dest = evt.get("dest", "")

    def is_private(ip):
        return (
            ip.startswith("10.") or
            ip.startswith("192.168.") or
            ip.startswith("172.16.") or
            ip.startswith("172.17.") or
            ip.startswith("172.18.") or
            ip.startswith("172.19.") or
            ip.startswith("172.20.") or
            ip.startswith("172.21.") or
            ip.startswith("172.22.") or
            ip.startswith("172.23.") or
            ip.startswith("172.24.") or
            ip.startswith("172.25.") or
            ip.startswith("172.26.") or
            ip.startswith("172.27.") or
            ip.startswith("172.28.") or
            ip.startswith("172.29.") or
            ip.startswith("172.30.") or
            ip.startswith("172.31.")
        )

    src_private = 1.0 if is_private(src) else 0.0
    dest_private = 1.0 if is_private(dest) else 0.0

    # deviation from expected user login hour
    user = evt.get("user", "unknown")
    prof = user_profile.get(user, {"mean_hour": 12, "std_hour": 4})

    mean_h = prof["mean_hour"]
    std_h = prof["std_hour"] if prof["std_hour"] > 0 else 1.0

    hour_dev = abs(hour - mean_h)
    hour_z = hour_dev / (std_h + 0.1)

    return np.array([
        hour_bin,
        success_flag,
        sig,
        src_private,
        dest_private,
        hour_z
    ], dtype="float32")