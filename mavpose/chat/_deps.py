"""Check that the optional chat dependencies are installed."""

from __future__ import annotations

import importlib.util

# import name -> pip distribution name
CHAT_DEPENDENCIES = {
    "langchain_openai": "langchain-openai",
    "langchain_core": "langchain-core",
    "langchain_chroma": "langchain-chroma",
    "chromadb": "chromadb",
    "openai": "openai",
    "dotenv": "python-dotenv",
    "matplotlib": "matplotlib",
}

INSTALL_HINT = "pip install 'mavpose[chat]'"


class ChatDependenciesMissing(ImportError):
    """Raised when the chat assistant is used without the [chat] extra."""


def missing_chat_dependencies() -> list:
    """Return the pip names of chat dependencies that are not installed."""
    return [
        dist for module, dist in CHAT_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]


def require_chat_deps() -> None:
    """Raise a helpful error if any chat dependency is missing."""
    missing = missing_chat_dependencies()
    if missing:
        raise ChatDependenciesMissing(
            "The MAVPose chat assistant needs extra packages that are not "
            f"installed ({', '.join(missing)}). Install them with:\n\n"
            f"    {INSTALL_HINT}\n\n"
            "The core data layer (mavpose.LogExtractor) works without them."
        )
