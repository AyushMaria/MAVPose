"""
mavpose.chat — natural-language plot assistant built on the MAVPose core.

Requires the optional extra:  pip install 'mavpose[chat]'

This package only uses the core's public API (mavpose.LogExtractor,
mavpose.validate_mavlink_file); it is never imported by the core.
"""

from __future__ import annotations

__all__ = ["PlotCreator", "execute_script"]


def __getattr__(name):
    if name == "PlotCreator":
        from mavpose.chat.plot_creator import PlotCreator
        return PlotCreator
    if name == "execute_script":
        from mavpose.chat.safe_executor import execute_script
        return execute_script
    raise AttributeError(f"module 'mavpose.chat' has no attribute {name!r}")
