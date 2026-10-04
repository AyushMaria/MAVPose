"""Deprecated location; use `mavpose.chat.plot_creator`. Will be removed in 0.3."""

import warnings

from mavpose.chat.plot_creator import PlotCreator  # noqa: F401

warnings.warn(
    "mavpose.PlotCreator has moved to mavpose.chat.plot_creator",
    DeprecationWarning,
    stacklevel=2,
)
