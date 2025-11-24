# ml_sidecar/config_loader.py
import yaml

def load_settings(path="./config/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)