from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents (main model and judge).

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

# Common misspellings / alternative names -> canonical provider name.
PROVIDER_ALIASES = {
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "claude": "anthropic",
    "google": "gemini",
    "google_genai": "gemini",
    "google-genai": "gemini",
    "openai_compatible": "custom",
    "openai-compatible": "custom",
    "open_router": "openrouter",
    "open-router": "openrouter",
}


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic` and reject unknown providers."""

    key = (value or "").strip().lower()
    provider = PROVIDER_ALIASES.get(key, key)
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {value!r}. Expected one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return provider


def has_credentials(config: ProviderConfig) -> bool:
    """True when the live path has a chance to work (key present, or local Ollama)."""

    provider = normalize_provider(config.provider)
    if provider == "ollama":
        return True
    if provider == "custom":
        return bool(config.base_url)
    return bool(config.api_key)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Imports are done lazily inside each branch so offline mode and tests never
    depend on every provider SDK being importable.
    """

    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        if provider == "custom" and not config.base_url:
            raise ValueError("Provider 'custom' requires a base_url (set CUSTOM_BASE_URL).")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url,
        )

    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(
        model=config.model_name,
        temperature=config.temperature,
        api_key=config.api_key,
    )


def read_live_turn(result: dict) -> tuple[str, int | None, int | None]:
    """Extract (reply text, input tokens, output tokens) for the latest turn of an agent run.

    Token counts come from provider `usage_metadata`; they are None when the
    provider does not report usage, so callers can fall back to estimates.
    """

    messages = result.get("messages", [])
    last_human = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"), default=-1)
    turn = [m for m in messages[last_human + 1 :] if getattr(m, "type", "") == "ai"]
    reply = turn[-1] if turn else (messages[-1] if messages else None)
    text = (getattr(reply, "text", None) or str(getattr(reply, "content", ""))) if reply else ""

    usages = [m.usage_metadata for m in turn if getattr(m, "usage_metadata", None)]
    if not usages:
        return text, None, None
    return text, sum(u.get("input_tokens", 0) for u in usages), sum(u.get("output_tokens", 0) for u in usages)
