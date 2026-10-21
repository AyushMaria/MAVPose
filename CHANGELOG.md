# Changelog

All notable changes to MAVPose. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/) as described in
[API stability](docs/api-stability.md).

## [0.2.0] — unreleased

The core is now a standalone, trustworthy data layer for ArduPilot and PX4
logs. The plot assistant is an optional extra on top.

### Breaking
- `pip install mavpose` installs the data layer only. Install
  `mavpose[chat]` for the plot assistant and the `mavpose` command.
- Numeric columns hold **converted units** (e.g. `GLOBAL_POSITION_INT.alt`
  is metres, not millimetres). Pass `convert_units=False` for raw values.
- **"Unknown" markers become NaN** (e.g. `SYS_STATUS.current_battery = -1`).
  Pass `filter_unknown=False` to keep them.
- `time_s` is now one shared clock for all message types. In 0.1 it could
  be a message counter (.bin) or mix several clocks (.tlog).

### Added
- `mavpose.load(path) -> FlightLog`: `messages()`, `series()` (indexed by
  `time_s`, unit in `attrs`), `unit()`, `frame()`, `summary()`,
  `to_parquet()`, `unknown_counts`, `stats`, `duration_s`.
- PX4 ULog (`.ulg`) support via pyulog, with units and unknown markers from
  PX4's message definitions.
- Units for every column (`LogExtractor.units`, `raw_units`).
- `LogExtractor.stats` and `unknown_counts`; corrupt packets are skipped and
  counted.
- Documentation: guide, format notes, API reference and stability policy.

### Fixed
- Generated plot scripts now actually run in the restricted executor; the
  executor blocks process spawning, network access and out-of-folder writes,
  and no longer exposes API keys.
- One corrupt packet no longer silently truncates the rest of a log.
- CI lints and tests the real package on Python 3.10–3.13, plus a core-only
  install.
- Installs on Python 3.12+ (matplotlib pin relaxed).

### Deprecated
- `mavpose.PlotCreator`, `mavpose.safe_executor` and `mavpose.cli` module
  paths. Use `mavpose.chat.*`. They will be removed in 0.3.

### Removed
- Committed build output (`site/`), the stale `target/` folder and the
  `app.py` stub.

## [0.1.0]

First release on PyPI.
