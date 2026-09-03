"""Shared helpers for eval scripts: load secrets from the repo-root .env.

Avoids hardcoding API keys in scripts. Zero external dependencies.

Usage:
    from env_utils import get_api_key
    KEY = get_api_key()                      # LLAMA_GATEWAY_API_KEY by default
    KEY = get_api_key("NEURALWATT_API_KEY")   # any other variable

Values already present in the environment take precedence over .env.
"""
import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env() -> None:
    """Populate os.environ from the repo-root .env (existing vars win)."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key(name: str = "LLAMA_GATEWAY_API_KEY") -> str:
    """Return the API key for `name`, from env vars or the repo .env file."""
    load_env()
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}: set it in {ENV_PATH} (see .env.example) "
            "or export it before running."
        )
    return value
