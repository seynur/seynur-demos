from datetime import datetime

def parse_windows_time(ts: str):
    """
    Parses timestamps like:
        2025-11-18T08:05:19.844741Z
    Fixes:
      - Z timezone
      - microseconds longer than 6 digits
    """
    if not ts:
        return None

    try:
        # Convert trailing Z → +00:00
        ts = ts.replace("Z", "+00:00")

        # Fix long digit microseconds (Python allows max 6)
        if "." in ts:
            date, rest = ts.split(".")
            micros, zone = rest.split("+")
            micros = micros[:6]  # trim microseconds
            ts = f"{date}.{micros}+{zone}"

        return datetime.fromisoformat(ts)
    except Exception:
        return None