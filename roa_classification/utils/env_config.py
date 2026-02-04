"""Load environment configuration from .env file."""

import os
from pathlib import Path
from functools import lru_cache


def _load_dotenv(env_path: Path) -> dict:
    """Parse a .env file into a dictionary, supporting ${VAR} interpolation."""
    values = {}
    if not env_path.exists():
        return values
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Simple variable interpolation
            for k, v in values.items():
                val = val.replace(f"${{{k}}}", v)
            values[key] = val
    return values


@lru_cache(maxsize=1)
def get_env_config() -> dict:
    """Load and cache .env configuration.

    Searches for .env starting from this file's location upward.
    """
    search = Path(__file__).resolve().parent
    while search != search.parent:
        env_path = search / ".env"
        if env_path.exists():
            return _load_dotenv(env_path)
        search = search.parent
    return {}


def get_data_dir() -> Path:
    cfg = get_env_config()
    return Path(
        cfg.get(
            "DATA_DIR",
            "/common/users/shared/pracsys/genMoPlan/data_trajectories",
        )
    )


def get_exp_dir() -> Path:
    cfg = get_env_config()
    return Path(
        cfg.get(
            "EXP_DIR",
            "/common/users/shared/pracsys/adaptive_roa_experiments",
        )
    )


def get_net_id() -> str:
    cfg = get_env_config()
    return cfg.get("NET_ID", os.environ.get("USER", "unknown"))
