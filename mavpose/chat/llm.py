"""
mavpose/chat/llm.py

Choose and build the language model used by the plot assistant.

Any of these work:

=============  ==========================================  ======================
Provider       Key variable                                Notes
=============  ==========================================  ======================
openrouter     ``OPENROUTER_API_KEY``                      default model ``z-ai/glm-5.1``
openai         ``OPENAI_API_KEY``
anthropic      ``ANTHROPIC_API_KEY``                       native client
gemini         ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``)  OpenAI-compatible endpoint
groq           ``GROQ_API_KEY``
deepseek       ``DEEPSEEK_API_KEY``
mistral        ``MISTRAL_API_KEY``
together       ``TOGETHER_API_KEY``
ollama         none (local)                                http://localhost:11434
lmstudio       none (local)                                http://localhost:1234
openai-compatible  ``MAVPOSE_API_KEY`` (optional)          needs ``MAVPOSE_BASE_URL``
=============  ==========================================  ======================

How the model is chosen, first match wins:

1. ``--model`` / the ``model`` argument, as ``provider:model``
   (e.g. ``anthropic:<model>``, ``ollama:llama3.1:8b``).
2. ``MAVPOSE_MODEL``, same format.
3. ``OPENROUTER_API_KEY`` is set: OpenRouter, with ``OPENROUTER_MODEL`` or the
   default model (this is how MAVPose 0.2 was configured).
4. Exactly one of the other key variables is set: that provider.  A model
   name is then still needed, because MAVPose does not guess model names
   that providers retire.

``MAVPOSE_BASE_URL`` overrides the endpoint of the chosen provider (for a
proxy or a remote Ollama) and ``MAVPOSE_API_KEY`` overrides its key.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from typing import Mapping, Optional

__all__ = [
    "ModelConfigError",
    "MissingAPIKeyError",
    "ModelConfig",
    "PROVIDERS",
    "resolve_model",
    "build_chat_model",
]


class ModelConfigError(RuntimeError):
    """The plot assistant's model could not be configured."""


class MissingAPIKeyError(ModelConfigError):
    """No API key was found for the chosen (or any) provider."""


@dataclass(frozen=True)
class _Provider:
    key_vars: tuple          # environment variables holding the key, in order
    base_url: Optional[str]  # None: the client library's default endpoint
    local: bool = False      # runs on this machine; no key needed
    default_model: Optional[str] = None
    models_url: str = ""     # where to look up model names


PROVIDERS: dict = {
    "openrouter": _Provider(("OPENROUTER_API_KEY",), "https://openrouter.ai/api/v1",
                            default_model="z-ai/glm-5.1",
                            models_url="https://openrouter.ai/models"),
    "openai": _Provider(("OPENAI_API_KEY",), None,
                        models_url="https://platform.openai.com/docs/models"),
    "anthropic": _Provider(("ANTHROPIC_API_KEY",), None,
                           models_url="https://docs.claude.com/en/docs/about-claude/models/overview"),
    "gemini": _Provider(("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                        "https://generativelanguage.googleapis.com/v1beta/openai/",
                        models_url="https://ai.google.dev/gemini-api/docs/models"),
    "groq": _Provider(("GROQ_API_KEY",), "https://api.groq.com/openai/v1",
                      models_url="https://console.groq.com/docs/models"),
    "deepseek": _Provider(("DEEPSEEK_API_KEY",), "https://api.deepseek.com",
                          models_url="https://api-docs.deepseek.com/"),
    "mistral": _Provider(("MISTRAL_API_KEY",), "https://api.mistral.ai/v1",
                         models_url="https://docs.mistral.ai/getting-started/models/"),
    "together": _Provider(("TOGETHER_API_KEY",), "https://api.together.xyz/v1",
                          models_url="https://docs.together.ai/docs/serverless-models"),
    "ollama": _Provider((), "http://localhost:11434/v1", local=True,
                        models_url="run `ollama list`"),
    "lmstudio": _Provider((), "http://localhost:1234/v1", local=True,
                          models_url="the model list in LM Studio"),
    "openai-compatible": _Provider(("MAVPOSE_API_KEY",), None, local=True,
                                   models_url="your server's documentation"),
}

# Providers that are picked automatically when only their key is set.
_AUTODETECT = [p for p, cfg in PROVIDERS.items() if cfg.key_vars and p not in
               ("openrouter", "openai-compatible")]


@dataclass(frozen=True)
class ModelConfig:
    """The resolved model: which provider, which model, and how to reach it."""

    provider: str
    model: str
    api_key: Optional[str]
    base_url: Optional[str]

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def _get(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def _split(spec: str) -> tuple:
    """'provider:model' -> (provider, model).  A bare name has no provider."""
    spec = spec.strip()
    head, sep, tail = spec.partition(":")
    if sep and head.lower() in PROVIDERS:
        return head.lower(), tail.strip()
    return None, spec


def _setup_help() -> str:
    lines = ["Set the API key for your provider, and choose a model with",
             "--model provider:model or MAVPOSE_MODEL=provider:model. For example:",
             "  OPENAI_API_KEY=...      MAVPOSE_MODEL=openai:<model>",
             "  ANTHROPIC_API_KEY=...   MAVPOSE_MODEL=anthropic:<model>",
             "  GEMINI_API_KEY=...      MAVPOSE_MODEL=gemini:<model>",
             "  OPENROUTER_API_KEY=...  (model optional; default z-ai/glm-5.1)",
             "  no key, local Ollama:   MAVPOSE_MODEL=ollama:llama3.1:8b",
             "Put these in a .env file in the folder you run mavpose from, or set",
             "them in your shell. Providers: " + ", ".join(PROVIDERS) + "."]
    return "\n".join(lines)


def resolve_model(model: Optional[str] = None,
                  env: Optional[Mapping[str, str]] = None) -> ModelConfig:
    """
    Work out which provider and model to use (see the module docstring).

    Parameters
    ----------
    model:
        ``provider:model``, or a bare model name for the auto-detected
        provider.  ``None`` uses ``MAVPOSE_MODEL`` or auto-detection.
    env:
        Variables to read; defaults to ``os.environ``.

    Raises
    ------
    MissingAPIKeyError
        No usable key for the chosen provider, or no provider configured.
    ModelConfigError
        Unknown provider, missing model name, or a missing base URL.
    """
    env = os.environ if env is None else env
    spec = (model or "").strip() or _get(env, "MAVPOSE_MODEL")
    provider, name = _split(spec) if spec else (None, "")

    if spec and provider is None and ":" in spec:
        # "anthropc:..." is a typo; "llama3.1:8b" is a bare model name.
        head = spec.split(":", 1)[0].lower()
        close = difflib.get_close_matches(head, list(PROVIDERS), n=1, cutoff=0.75)
        if close:
            raise ModelConfigError(
                f"Unknown provider {head!r}. Did you mean {close[0]!r}? "
                f"Providers: {', '.join(PROVIDERS)}.")

    if provider is None:
        provider = _detect_provider(env)
    if not name and provider == "openrouter":
        name = _get(env, "OPENROUTER_MODEL")

    cfg = PROVIDERS[provider]
    if not name:
        name = cfg.default_model or ""
    if not name:
        raise ModelConfigError(
            f"Choose which {provider} model to use: --model {provider}:<model-name> or "
            f"MAVPOSE_MODEL={provider}:<model-name>. Model names: {cfg.models_url}")

    api_key = _get(env, "MAVPOSE_API_KEY") or next(
        (v for v in (_get(env, k) for k in cfg.key_vars) if v), "")
    if not api_key and not cfg.local:
        names = " or ".join(cfg.key_vars)
        raise MissingAPIKeyError(f"{names} is not set (needed for {provider}).\n" + _setup_help())

    base_url = _get(env, "MAVPOSE_BASE_URL") or cfg.base_url
    if provider == "openai-compatible" and not base_url:
        raise ModelConfigError(
            "openai-compatible needs MAVPOSE_BASE_URL, e.g. http://my-server:8000/v1")

    return ModelConfig(provider, name, api_key or None, base_url)


def _detect_provider(env: Mapping[str, str]) -> str:
    if _get(env, "OPENROUTER_API_KEY"):
        return "openrouter"
    found = [p for p in _AUTODETECT if any(_get(env, k) for k in PROVIDERS[p].key_vars)]
    if len(found) == 1:
        return found[0]
    if not found:
        raise MissingAPIKeyError("No AI provider is configured.\n" + _setup_help())
    raise ModelConfigError(
        f"Keys for several providers are set ({', '.join(found)}). Choose one with "
        f"--model provider:model or MAVPOSE_MODEL, e.g. {found[0]}:<model-name>.")


def build_chat_model(config: ModelConfig, max_tokens: int, temperature: float = 0):
    """Create the LangChain chat model for *config*."""
    if config.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - part of the [chat] extra
            raise ModelConfigError(
                "Anthropic models need the langchain-anthropic package: "
                'pip install -U "mavpose[chat]"') from exc
        kwargs = dict(model=config.model, max_tokens=max_tokens,
                      temperature=temperature, api_key=config.api_key)
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatAnthropic(**kwargs)

    from langchain_openai import ChatOpenAI

    kwargs = dict(model=config.model, max_tokens=max_tokens, temperature=temperature,
                  # Local servers ignore the key, but the client requires one.
                  api_key=config.api_key or "not-needed")
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenAI(**kwargs)
