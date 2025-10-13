import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from aisuite import Client

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")

_client: Client | None = None


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} placeholders using environment variables."""
    if isinstance(value, str):
        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                return ""
            return value

        return ENV_PATTERN.sub(replacer, value)

    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_expand_env(item) for item in value]

    return value


def _load_config_from_env() -> Dict[str, Any]:
    """Load provider config from the AISUITE_CONFIG environment variable."""
    config_path = os.getenv("AISUITE_CONFIG")
    if not config_path:
        return {}

    path = Path(config_path).expanduser()
    if not path.is_file():
        return {}

    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text())

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load AISUITE_CONFIG "
            f"({path}) but is not installed."
        )

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_config_from_default_locations() -> Dict[str, Any]:
    """Search for `.aisuite.yaml` in common locations."""
    candidate_paths = [
        Path(__file__).resolve().parents[1] / ".aisuite.yaml",
        Path.home() / ".aisuite.yaml",
    ]

    for path in candidate_paths:
        if not path.is_file():
            continue

        if yaml is None:
            # Skip silently so environments without PyYAML can still work.
            return {}

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    return {}


def _load_provider_configs() -> Dict[str, Any]:
    """Load provider configuration from env or default config files."""
    config = _load_config_from_env()
    if not config:
        config = _load_config_from_default_locations()

    providers = config.get("providers") if isinstance(config, dict) else None
    return providers or {}


def get_aisuite_client() -> Client:
    """Return a singleton aisuite Client configured with provider settings."""
    global _client

    if _client is None:
        provider_configs = _load_provider_configs()
        provider_configs = _expand_env(provider_configs)
        _client = Client(provider_configs=provider_configs)

    return _client
