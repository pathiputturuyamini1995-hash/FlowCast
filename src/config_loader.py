from pathlib import Path
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config = load_config()
    print(config)