"""Deprecated location; use `mavpose.chat.safe_executor`. Will be removed in 0.3."""

import warnings

from mavpose.chat.safe_executor import execute_script, build_runner, BLOCKED_MODULES  # noqa: F401

warnings.warn(
    "mavpose.safe_executor has moved to mavpose.chat.safe_executor",
    DeprecationWarning,
    stacklevel=2,
)
