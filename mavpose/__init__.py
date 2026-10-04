"""
MAVPose — clean, unit-correct telemetry from drone flight logs.

Core (``pip install mavpose``): parse MAVLink .tlog and ArduPilot DataFlash
.bin/.log files into time-aligned DataFrames with real units.

    from mavpose import LogExtractor
    frames = LogExtractor("flight.tlog").extract_all()

Chat assistant (``pip install 'mavpose[chat]'``): plain-English plots,
available as ``mavpose.chat.PlotCreator`` and the ``mavpose`` command.
"""

from mavpose.file_validator import FileValidationError, validate_mavlink_file
from mavpose.log_extractor import LogExtractor

__all__ = ["LogExtractor", "validate_mavlink_file", "FileValidationError", "PlotCreator"]
__version__ = "0.1.0"


def __getattr__(name):
    # Kept for backwards compatibility: `from mavpose import PlotCreator`.
    # Imported lazily so the core never loads the chat dependencies.
    if name == "PlotCreator":
        from mavpose.chat.plot_creator import PlotCreator
        return PlotCreator
    raise AttributeError(f"module 'mavpose' has no attribute {name!r}")
