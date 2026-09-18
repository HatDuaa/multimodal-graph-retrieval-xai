"""Load the shared YAML config and resolve paths against the repository root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path for an entry of cfg['paths']; the directory is created if missing."""
    p = REPO_ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
