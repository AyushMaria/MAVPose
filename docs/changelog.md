# Changelog

All notable changes to MAVPose. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/) as described in
[API stability](api-stability.md).

## [0.3.0] — unreleased

### Breaking
- **The `mavpose` command no longer runs generated code by default.** The
  model now describes the plot as a JSON [plot spec](guide/chat.md#plot-specs), which
  MAVPose checks against the log and draws itself. Pass `--code` for the
  previous behaviour (a generated script run in the restricted subprocess).
  `PlotCreator` takes `mode="spec"` (default) or `mode="code"`.

### Added
- **Plot specs**: plots are saved as `plot.json` next to `plot.png`, and
  `mavpose LOG --spec plot.json` redraws a spec on any log with no model or
  API key. Invalid specs (e.g. a misspelt field) go back to the model with
  the exact error. Axis labels and legends get units from the log.
  `PlotCreator.plot_from_spec()`, `last_spec`, `spec_path`, and
  `mavpose.chat.plotspec` (`render_spec`, `validate_spec`, `parse_spec`,
  `PlotSpecError`).
- **The plot assistant works with any AI provider**: OpenAI, Anthropic,
  Google Gemini, OpenRouter, Groq, DeepSeek, Mistral, Together, local Ollama
  and LM Studio, or any OpenAI-compatible server. Choose with
  `--model provider:model` or `MAVPOSE_MODEL`; if only one provider's key is
  set, it's used automatically. `MAVPOSE_BASE_URL` and `MAVPOSE_API_KEY`
  point any provider at another address. See
  [Choosing an AI provider](guide/chat.md#choosing-an-ai-provider).
- `PlotCreator(model=...)`, `PlotCreator.model_config`, and
  `mavpose.chat.ModelConfigError` (the parent of `MissingAPIKeyError`).

### Changed
- **The model now picks the message types** a request needs from the log's
  own list, instead of a vector search. Broad requests ("compare GPS and
  barometer altitude with battery voltage") no longer miss types, and no
  embeddings API is needed, which is what made other providers possible.
- Setups with only `OPENROUTER_API_KEY` (and optionally `OPENROUTER_MODEL`)
  work exactly as before.

### Removed
- ChromaDB and `langchain-chroma` are no longer dependencies of
  the `[chat]` extra; `langchain-anthropic` is added. `pip install
  "mavpose[chat]"` now installs about 60 packages instead of 115.
- `PlotCreator.db` (the vector store) is gone.

## [0.2.1] — 2026-10-21

Fixes for the plot assistant (`mavpose[chat]`). The core library and the
MCP server are unchanged.

### Fixed
- **"Plot saved" no longer appears when the plot failed.** A run only
  counts as successful when the script saves a new `plot.png`; the old one
  is deleted first, so a plot from an earlier run can't pass for a new one.
  A script that exits cleanly without saving the figure is sent back to the
  model to fix. With `--prompt`, the command now exits with status 1 when no
  plot was created. `PlotCreator.last_success` reports the outcome.
- **Logs in the same folder no longer share a search index.** The index of
  message types is now kept in memory, one per log, instead of in a
  `chroma_db/` folder next to the log that grew on every run and mixed
  message types from different logs. You can delete old `chroma_db/`
  folders.
- **A missing API key gives a clear message** instead of a `KeyError`
  traceback, and an empty `OPENROUTER_API_KEY=` counts as missing
  (`PlotCreator` raises `MissingAPIKeyError`).
- **`.env` is found when MAVPose is installed with pip.** It is now read
  from the folder you run `mavpose` in (or a parent folder). Before, only
  a `.env` next to the MAVPose source was found, so the documented setup
  only worked from a clone of the repository.
- Leftover "MAVPlot" name in `template.env`.

## [0.2.0] — 2026-10-21

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
- Requires **pymavlink 2.4.48 or newer** (was 2.4.37). Older versions
  can't read DataFlash unit records or recover from corrupt `.tlog` data.

### Added
- `mavpose-mcp`, an MCP server (`pip install "mavpose[mcp]"`) that gives AI
  agents seven tools: `list_logs`, `summarize_flight`, `list_messages`,
  `describe_message`, `get_series`, `get_log_messages` and `export_parquet`.
  It's read-only except export, limited to allowed folders, and runs no code.
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
