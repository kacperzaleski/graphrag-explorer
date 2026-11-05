"""Provider switching is pure config logic, so it is tested without any network calls."""

import pytest

from backend.config import Settings


def make(**env) -> Settings:
    # _env_file=None: ignore the developer's real .env so tests are deterministic
    return Settings(_env_file=None, **env)


def test_gemini_defaults():
    s = make(gemini_api_key="g-key")
    chat, embed = s.chat_endpoint(), s.embed_endpoint()
    assert chat.provider == "gemini" and chat.api_key == "g-key"
    assert embed.model == "gemini-embedding-001" and embed.dimensions == 768


def test_switch_chat_provider_keeps_embeddings():
    s = make(llm_provider="mistral", mistral_api_key="m-key", gemini_api_key="g-key")
    assert s.chat_endpoint().base_url == "https://api.mistral.ai/v1"
    assert s.embed_endpoint().provider == "gemini"


def test_model_override():
    s = make(llm_provider="cerebras", llm_model="llama-3.3-70b", cerebras_api_key="c")
    assert s.chat_endpoint().model == "llama-3.3-70b"


def test_generic_key_fallback():
    assert make(llm_api_key="fallback").chat_endpoint().api_key == "fallback"


def test_missing_key_is_a_clear_error():
    with pytest.raises(RuntimeError, match="CEREBRAS_API_KEY"):
        make(llm_provider="cerebras").chat_endpoint()


def test_provider_without_embeddings_is_rejected():
    with pytest.raises(ValueError, match="no embeddings API"):
        make(embed_provider="cerebras", cerebras_api_key="c").embed_endpoint()


def test_fixed_size_embeddings_send_no_dimensions():
    s = make(embed_provider="mistral", mistral_api_key="m")
    assert s.embed_endpoint().dimensions is None and s.embed_dim() == 1024


def test_secrets_are_not_printed():
    assert "g-key" not in repr(make(gemini_api_key="g-key"))


def test_answer_model_defaults_to_extraction_model():
    s = make(gemini_api_key="g")
    assert s.chat_endpoint("answer") == s.chat_endpoint("extract")


def test_separate_answer_provider():
    s = make(gemini_api_key="g", cerebras_api_key="c", answer_provider="cerebras")
    assert s.chat_endpoint("extract").provider == "gemini"
    assert s.chat_endpoint("answer").provider == "cerebras"


def test_judge_can_use_another_provider():
    s = make(gemini_api_key="g", groq_api_key="q", judge_provider="groq")
    assert s.chat_endpoint("judge").provider == "groq"
    assert s.chat_endpoint("answer").provider == "gemini"
