from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from memory_store import DEFAULT_MIN_CONFIDENCE
from model_provider import ProviderConfig, normalize_provider

DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}
DEFAULT_COMPACT_THRESHOLD_TOKENS = 500
DEFAULT_COMPACT_KEEP_MESSAGES = 4

# provider -> (env vars holding the API key, env var holding the base URL)
PROVIDER_ENV = {
    "openai": (("OPENAI_API_KEY",), None),
    "custom": (("CUSTOM_API_KEY",), "CUSTOM_BASE_URL"),
    "gemini": (("GEMINI_API_KEY", "GOOGLE_API_KEY"), None),
    "anthropic": (("ANTHROPIC_API_KEY",), None),
    "ollama": ((), "OLLAMA_BASE_URL"),
    "openrouter": (("OPENROUTER_API_KEY",), None),
}
DEFAULT_BASE_URLS = {"ollama": "http://localhost:11434"}


def _env(name: str, default: str | None = None) -> str | None:
    """Read an env var, treating empty strings as missing."""

    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _provider_from_env(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Build a ProviderConfig from `<PREFIX>_PROVIDER`, `<PREFIX>_MODEL`, `<PREFIX>_TEMPERATURE`.

    Missing values fall back to `fallback` (used so the judge reuses the main model),
    then to the lab defaults. Missing API keys are allowed: offline mode needs none.
    """

    provider = normalize_provider(
        _env(f"{prefix}_PROVIDER", fallback.provider if fallback else DEFAULT_PROVIDER)
    )
    same_provider = fallback is not None and fallback.provider == provider
    model_name = _env(
        f"{prefix}_MODEL", fallback.model_name if same_provider else DEFAULT_MODELS[provider]
    )
    temperature = float(
        _env(f"{prefix}_TEMPERATURE", str(fallback.temperature) if fallback else "0.0")
    )

    key_vars, base_url_var = PROVIDER_ENV[provider]
    api_key = next((_env(var) for var in key_vars if _env(var)), None)
    base_url = _env(base_url_var, DEFAULT_BASE_URLS.get(provider)) if base_url_var else None

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - Paths: repo root, dataset directory, state directory.
    - Compact memory: token threshold and number of recent messages kept verbatim.
    - Providers for the main model and the judge (`openai`, `custom`, `gemini`,
      `anthropic`, `ollama`, `openrouter`).
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    # Only facts at or above this confidence are written to User.md (bonus: confidence threshold).
    min_fact_confidence: float = DEFAULT_MIN_CONFIDENCE


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` / environment variables and return a populated LabConfig.

    1. Resolve the repo root (defaults to the parent of `src/`).
    2. Load values from `<root>/.env` when present.
    3. Create `state/` if it does not exist.
    4. Missing API keys are allowed: offline mode needs none.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_from_env("LLM")
    judge_model = _provider_from_env("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(
            _env("COMPACT_THRESHOLD_TOKENS", str(DEFAULT_COMPACT_THRESHOLD_TOKENS))
        ),
        compact_keep_messages=int(
            _env("COMPACT_KEEP_MESSAGES", str(DEFAULT_COMPACT_KEEP_MESSAGES))
        ),
        model=model,
        judge_model=judge_model,
        min_fact_confidence=float(_env("MIN_FACT_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE))),
    )
