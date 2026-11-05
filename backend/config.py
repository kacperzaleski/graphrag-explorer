"""Central settings, read from environment variables / .env (pydantic-settings).

Switching LLM provider is a one-line change in .env (LLM_PROVIDER=mistral). Every provider here
speaks the OpenAI-compatible API, so the same `openai` client works for all of them; only base URL,
key and model name differ. Those differences live in PROVIDERS below.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Provider:
    base_url: str
    chat_model: str
    embed_model: str | None = None  # None = this provider has no embeddings API
    embed_dim: int | None = None
    # Gemini lets you request a shorter vector ("Matryoshka" embeddings); Mistral's size is fixed.
    supports_dimensions: bool = False


# Defaults checked on 2026-11-05. Model names change often: override with LLM_MODEL / EMBED_MODEL.
PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        chat_model="gemini-3.5-flash-lite",
        embed_model="gemini-embedding-001",
        embed_dim=768,
        supports_dimensions=True,
    ),
    "cerebras": Provider(base_url="https://api.cerebras.ai/v1", chat_model="gpt-oss-120b"),
    "mistral": Provider(
        base_url="https://api.mistral.ai/v1",
        chat_model="mistral-small-latest",
        embed_model="mistral-embed",
        embed_dim=1024,
    ),
    "groq": Provider(base_url="https://api.groq.com/openai/v1", chat_model="openai/gpt-oss-120b"),
}


@dataclass(frozen=True)
class Endpoint:
    """Everything needed to call one model: resolved from provider preset + overrides + key."""

    provider: str
    base_url: str
    model: str
    api_key: str
    dimensions: int | None = None  # embeddings only: requested vector size, if the API accepts one


class Settings(BaseSettings):
    # .env found by path (not the current directory), so `python -m` works from anywhere
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    llm_provider: str = "gemini"
    llm_model: str | None = None  # None = provider's default chat model
    embed_provider: str = "gemini"
    embed_model: str | None = None

    # SecretStr: printing/logging settings shows '**********' instead of the key
    gemini_api_key: SecretStr | None = None
    cerebras_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    llm_api_key: SecretStr | None = None  # generic fallback

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("graphrag-dev-password")

    def _api_key(self, provider: str) -> str:
        key = getattr(self, f"{provider}_api_key") or self.llm_api_key
        if key is None or not key.get_secret_value():
            env_var = f"{provider.upper()}_API_KEY"
            raise RuntimeError(f"No API key for '{provider}': set {env_var} in .env")
        return key.get_secret_value()

    def chat_endpoint(self) -> Endpoint:
        preset = _preset(self.llm_provider)
        return Endpoint(
            self.llm_provider,
            preset.base_url,
            self.llm_model or preset.chat_model,
            self._api_key(self.llm_provider),
        )

    def embed_endpoint(self) -> Endpoint:
        preset = _preset(self.embed_provider)
        if preset.embed_model is None:
            raise ValueError(
                f"'{self.embed_provider}' has no embeddings API; "
                f"use EMBED_PROVIDER=gemini or mistral"
            )
        return Endpoint(
            self.embed_provider,
            preset.base_url,
            self.embed_model or preset.embed_model,
            self._api_key(self.embed_provider),
            dimensions=preset.embed_dim if preset.supports_dimensions else None,
        )

    def embed_dim(self) -> int:
        """Vector size. The Neo4j vector index is built with it: changing it means re-ingest."""
        return _preset(self.embed_provider).embed_dim


def _preset(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}', choose one of {sorted(PROVIDERS)}")
    return PROVIDERS[name]


@lru_cache
def get_settings() -> Settings:
    """One Settings instance per process (FastAPI will use this as a dependency in Phase 3)."""
    return Settings()
