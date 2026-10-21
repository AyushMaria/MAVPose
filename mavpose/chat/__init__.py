"""
mavpose.chat — natural-language plot assistant built on the MAVPose core.

Requires the optional extra:  pip install 'mavpose[chat]'

This package only uses the core's public API (mavpose.LogExtractor,
mavpose.validate_mavlink_file); it is never imported by the core.
"""

from __future__ import annotations

__all__ = ["PlotCreator", "MissingAPIKeyError", "ModelConfigError", "PlotSpecError",
           "render_spec", "execute_script"]


def __getattr__(name):
    if name == "PlotCreator":
        from mavpose.chat.plot_creator import PlotCreator
        return PlotCreator
    if name in ("MissingAPIKeyError", "ModelConfigError"):
        from mavpose.chat import llm
        return getattr(llm, name)
    if name in ("PlotSpecError", "render_spec"):
        from mavpose.chat import plotspec
        return getattr(plotspec, name)
    if name == "execute_script":
        from mavpose.chat.safe_executor import execute_script
        return execute_script
    raise AttributeError(f"module 'mavpose.chat' has no attribute {name!r}")
