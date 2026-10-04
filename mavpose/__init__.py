"""
MAVPose — clean, unit-correct telemetry from drone flight logs.

    import mavpose

    log = mavpose.load("flight.tlog")      # MAVLink .tlog, ArduPilot .bin/.log, PX4 .ulg
    log.messages()
    alt = log.series("GLOBAL_POSITION_INT", "alt")   # pandas Series indexed by time_s
    alt.attrs["unit"]                               # "m"

Public API (see docs/api-stability.md): ``load``, ``FlightLog``,
``LogExtractor``, ``validate_mavlink_file``, ``FileValidationError`` and
``__version__``.  The plain-English plot assistant lives in ``mavpose.chat``
and needs ``pip install 'mavpose[chat]'``.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from mavpose.file_validator import FileValidationError, validate_mavlink_file
from mavpose.flightlog import FlightLog, load
from mavpose.log_extractor import LogExtractor

try:
    __version__ = _version("mavpose")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0+unknown"

__all__ = [
    "load",
    "FlightLog",
    "LogExtractor",
    "validate_mavlink_file",
    "FileValidationError",
    "__version__",
]


def __getattr__(name):
    # Backwards compatibility: `from mavpose import PlotCreator` (deprecated
    # location, still supported).  Imported lazily so the core never loads the
    # chat dependencies.
    if name == "PlotCreator":
        from mavpose.chat.plot_creator import PlotCreator
        return PlotCreator
    raise AttributeError(f"module 'mavpose' has no attribute {name!r}")
