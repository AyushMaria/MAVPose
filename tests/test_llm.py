"""
tests/test_llm.py

Choosing the plot assistant's provider and model (mavpose.chat.llm).
resolve_model() takes the environment as a dict, so no real keys are needed
and nothing touches the network.
"""

import pytest

pytest.importorskip("langchain_openai", reason="needs the [chat] extra")

from mavpose.chat.llm import (  # noqa: E402
    PROVIDERS,
    MissingAPIKeyError,
    ModelConfigError,
    build_chat_model,
    resolve_model,
)


class TestBackwardCompatibility:
    """0.2 setups (OpenRouter only) keep working unchanged."""

    def test_openrouter_key_alone_uses_default_model(self):
        cfg = resolve_model(env={"OPENROUTER_API_KEY": "k"})
        assert (cfg.provider, cfg.model, cfg.api_key) == ("openrouter", "z-ai/glm-5.1", "k")
        assert cfg.base_url == "https://openrouter.ai/api/v1"

    def test_openrouter_model_variable_is_honoured(self):
        cfg = resolve_model(env={"OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "openai/gpt-4o"})
        assert str(cfg) == "openrouter:openai/gpt-4o"

    def test_openrouter_wins_when_other_keys_are_also_set(self):
        cfg = resolve_model(env={"OPENROUTER_API_KEY": "k", "OPENAI_API_KEY": "o"})
        assert cfg.provider == "openrouter"


class TestChoosingAProvider:

    @pytest.mark.parametrize("provider,key_var,base_url", [
        ("openai", "OPENAI_API_KEY", None),
        ("anthropic", "ANTHROPIC_API_KEY", None),
        ("gemini", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
        ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com"),
        ("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
        ("together", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
    ])
    def test_explicit_provider(self, provider, key_var, base_url):
        cfg = resolve_model(f"{provider}:some-model", env={key_var: "secret"})
        assert (cfg.provider, cfg.model, cfg.api_key, cfg.base_url) == (
            provider, "some-model", "secret", base_url)

    def test_mavpose_model_variable(self):
        cfg = resolve_model(env={"MAVPOSE_MODEL": "gemini:g-1", "GEMINI_API_KEY": "g"})
        assert str(cfg) == "gemini:g-1"

    def test_argument_beats_variable(self):
        cfg = resolve_model("openai:from-arg",
                            env={"MAVPOSE_MODEL": "gemini:from-env", "OPENAI_API_KEY": "o",
                                 "GEMINI_API_KEY": "g"})
        assert str(cfg) == "openai:from-arg"

    def test_google_api_key_works_for_gemini(self):
        assert resolve_model("gemini:g", env={"GOOGLE_API_KEY": "g"}).api_key == "g"

    def test_single_key_is_detected_and_bare_model_name_used(self):
        cfg = resolve_model("my-model", env={"ANTHROPIC_API_KEY": "a"})
        assert str(cfg) == "anthropic:my-model"

    def test_detected_provider_still_needs_a_model(self):
        with pytest.raises(ModelConfigError, match="Choose which openai model") as err:
            resolve_model(env={"OPENAI_API_KEY": "o"})
        assert "openai:<model-name>" in str(err.value)
        assert not isinstance(err.value, MissingAPIKeyError)

    def test_several_keys_need_a_choice(self):
        with pytest.raises(ModelConfigError, match="several providers"):
            resolve_model(env={"OPENAI_API_KEY": "o", "ANTHROPIC_API_KEY": "a"})

    def test_nothing_configured(self):
        with pytest.raises(MissingAPIKeyError, match="No AI provider is configured") as err:
            resolve_model(env={})
        assert "OPENAI_API_KEY" in str(err.value) and "ollama" in str(err.value)

    def test_chosen_provider_without_its_key(self):
        with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY is not set"):
            resolve_model("anthropic:x", env={"OPENAI_API_KEY": "o"})

    def test_blank_key_is_missing(self):
        with pytest.raises(MissingAPIKeyError):
            resolve_model("openai:x", env={"OPENAI_API_KEY": "   "})

    def test_typo_in_provider_is_caught(self):
        with pytest.raises(ModelConfigError, match="Did you mean 'anthropic'"):
            resolve_model("anthropc:x", env={"ANTHROPIC_API_KEY": "a"})


class TestLocalAndCustomEndpoints:

    def test_ollama_needs_no_key_and_keeps_colons_in_model_name(self):
        cfg = resolve_model("ollama:llama3.1:8b", env={})
        assert (cfg.model, cfg.api_key, cfg.base_url) == (
            "llama3.1:8b", None, "http://localhost:11434/v1")

    def test_lmstudio(self):
        assert resolve_model("lmstudio:m", env={}).base_url == "http://localhost:1234/v1"

    def test_bare_ollama_style_name_is_a_model_not_a_provider(self):
        cfg = resolve_model("llama3.1:8b", env={"GROQ_API_KEY": "q"})
        assert str(cfg) == "groq:llama3.1:8b"

    def test_base_url_override(self):
        cfg = resolve_model("ollama:m", env={"MAVPOSE_BASE_URL": "http://gpu-box:11434/v1"})
        assert cfg.base_url == "http://gpu-box:11434/v1"

    def test_api_key_override(self):
        cfg = resolve_model("openai:m", env={"MAVPOSE_API_KEY": "proxy-key"})
        assert cfg.api_key == "proxy-key"

    def test_openai_compatible_needs_a_base_url(self):
        with pytest.raises(ModelConfigError, match="MAVPOSE_BASE_URL"):
            resolve_model("openai-compatible:m", env={})
        cfg = resolve_model("openai-compatible:m", env={"MAVPOSE_BASE_URL": "http://h:8000/v1"})
        assert (cfg.base_url, cfg.api_key) == ("http://h:8000/v1", None)


class TestBuildChatModel:

    def test_openai_compatible_providers_use_their_endpoint(self):
        llm = build_chat_model(resolve_model("groq:m", env={"GROQ_API_KEY": "q"}), max_tokens=100)
        assert type(llm).__name__ == "ChatOpenAI"
        assert llm.openai_api_base == "https://api.groq.com/openai/v1"
        assert llm.model_name == "m" and llm.max_tokens == 100
        assert llm.openai_api_key.get_secret_value() == "q"

    def test_openai_uses_the_default_endpoint(self):
        llm = build_chat_model(resolve_model("openai:m", env={"OPENAI_API_KEY": "o"}), 10)
        assert llm.openai_api_base in (None, "")

    def test_local_model_gets_a_placeholder_key(self):
        llm = build_chat_model(resolve_model("ollama:m", env={}), 10)
        assert llm.openai_api_key.get_secret_value()

    def test_anthropic_uses_the_native_client(self):
        pytest.importorskip("langchain_anthropic")
        llm = build_chat_model(resolve_model("anthropic:m", env={"ANTHROPIC_API_KEY": "a"}), 50)
        assert type(llm).__name__ == "ChatAnthropic"
        assert llm.model == "m" and llm.max_tokens == 50


def test_every_provider_documents_where_to_find_model_names():
    for name, cfg in PROVIDERS.items():
        assert cfg.models_url, name
