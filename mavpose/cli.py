"""Deprecated location; use `mavpose.chat.cli`. Will be removed in 0.3."""

import warnings

from mavpose.chat.cli import main  # noqa: F401

warnings.warn(
    "mavpose.cli has moved to mavpose.chat.cli",
    DeprecationWarning,
    stacklevel=2,
)
