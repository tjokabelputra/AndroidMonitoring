import yaml
from pathlib import Path

def load_config():
    BASE_DIR = Path(__file__).resolve().parent.parent
    config_path = BASE_DIR / "config.yaml"

    with open(config_path, "r") as p:
        config = yaml.safe_load(p)

    return config