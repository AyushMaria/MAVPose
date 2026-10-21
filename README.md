<div align="center">

# 🧭 MAVPose

**Ask questions about your drone flight in plain English. Get plots.**

*Inspired by the Log Pose from One Piece — it locks onto your flight data and charts a course through it.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/AyushMaria/MAVPose/actions/workflows/ci.yml/badge.svg)](https://github.com/AyushMaria/MAVPose/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/mavpose.svg)](https://pypi.org/project/mavpose/)

</div>

---

MAVPose reads ArduPilot and PX4 flight logs into clean, time-aligned pandas data with real units, and its optional assistant turns a plain-English request into a plot. The assistant works with any AI provider, including local models. The model never sees the raw log: it gets the log's message types and a precise column schema with units, and replies with a small JSON **plot spec** that MAVPose checks and draws itself, so no generated code runs. Specs are saved, so any plot can be redrawn later without a model.

```
$ mavpose flight.tlog --model openai:<model> --prompt "Plot altitude over time"
🤖 Model: openai:<model>

📂 Parsing log schema: flight.tlog
✅ Found 38 message types.

🔍 Finding relevant message types for: 'Plot altitude over time'
   → ['GLOBAL_POSITION_INT', 'VFR_HUD']

🗃️  Extracting telemetry to Parquet...
┌──────────────────────────────────────────────────────────┐
│  🗃️  Extracted Parquet schema                               │
├──────────────────────────────────────────────────────────┤
│  [GLOBAL_POSITION_INT]  1842 rows                          │
│      time_s: float64 s  [0.0 … 312.4]                     │
│      alt: float64 m  [487.32 … 512.1]                     │
│  [VFR_HUD]  1842 rows                                      │
│      time_s: float64 s  [0.0 … 312.4]                     │
│      alt: float64 m  [476.1 … 501.3]                      │
└──────────────────────────────────────────────────────────┘
   Saved → /path/to/telemetry.parquet

✍️  Asking openai:<model> to design the plot...
✅ Plot saved to: /path/to/plot.png
   Spec saved to: /path/to/plot.json  (redraw with --spec)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  PHASE 1 — Extraction  (parent process, no LLM)             │
├──────────────────────────────────────────────────────────┤
│                                                             │
│  LogExtractor.schema_only()                                 │
│    └─ pymavlink fast scan → {msg_type: {fields, count}}     │
│                                                             │
│  find_relevant_data_types(prompt)                           │
│    └─ the model picks from the log's message types        │
│                                                             │
│  LogExtractor.extract_all()  ←  full row materialisation    │
│    └─ per-msg-type DataFrames, time_s column, cast numerics │
│                                                             │
│  LogExtractor.export_parquet(msg_types, path)               │
│    └─ clean telemetry.parquet + schema summary dict         │
│                                                             │
├──────────────────────────────────────────────────────────┤
│  PHASE 2 — LLM Plot Generation                              │
├──────────────────────────────────────────────────────────┤
│                                                             │
│  LLM receives:                                              │
│    └─ schema: {msg_type: {columns: {col: dtype, unit, range}}} │
│                                                             │
│  Default: LLM replies with a JSON plot spec                 │
│    {"panels": [{"series": [{"msg_type", "field"}]}]}       │
│    └─ MAVPose validates it against the log and draws it    │
│    └─ invalid spec? the error goes back to the LLM, N×     │
│    → plot.png + plot.json (redraw with --spec)             │
│                                                             │
│  --code: LLM writes a pandas + matplotlib script instead    │
│    └─ run in a restricted subprocess, fixed up to N×       │
│    → plot.png + plot.py                                    │
└──────────────────────────────────────────────────────────┘
```

**Why this matters:** In the old design, the LLM had to write `pymavlink` code to parse a binary `.tlog` — a notoriously tricky API with blocking/non-blocking subtleties, timestamp inconsistencies, and message-type guesswork. By the time the LLM touched the data, it was flying blind. Now it receives a typed, time-aligned Parquet file with exact column names, dtypes, and value ranges. Writing `pd.read_parquet()` + `matplotlib` code against a known schema is trivially reliable.

---

## Features

- **Plain-English plots** — describe any flight data; MAVPose figures out which MAVLink fields to use
- **ArduPilot and PX4** — one API for MAVLink `.tlog`, ArduPilot DataFlash `.bin`/`.log` and PX4 ULog `.ulg` logs
- **Headless extraction layer** — `LogExtractor` parses every supported format into per-message-type DataFrames before the LLM is ever invoked:
  - **one clock** — every message type shares a single `time_s` axis (tlog receive time, DataFlash `TimeUS`, or ULog `timestamp` from logging start); native time fields are kept as columns
  - **real units** — scaled fields are converted using unit metadata from the log or the autopilot's own message definitions (`alt` mm → m, `lat` degE7 → deg, `voltage_battery` mV → V, …) and every column's unit is reported; pass `convert_units=False` for raw values
  - **unknown values removed** — "not provided" markers (MAVLink `current_battery = -1`, `eph = UINT16_MAX`; PX4 `battery_status.current_a = -1`, …) become NaN instead of fake readings, using the `invalid` markers from the official MAVLink and PX4 definitions; counts appear in `extractor.unknown_counts` and the LLM schema. Pass `filter_unknown=False` to keep them
  - **corruption-tolerant** — bad packets are skipped and counted (`extractor.stats`) instead of silently ending the parse
- **Clean Parquet handoff** — only the relevant message types are exported; the LLM sees exact column names, dtypes, min/max ranges — no binary guesswork
- **Any AI provider** — OpenAI, Anthropic, Gemini, OpenRouter, Groq, DeepSeek, Mistral, Together, local Ollama / LM Studio, or any OpenAI-compatible server
- **Smart message selection** — the model picks the message types it needs from the log's own list
- **No generated code by default** — the model describes the plot as a JSON spec; MAVPose validates it against the log and draws it, with units on every axis. Invalid specs go back to the model with the exact error
- **Reproducible plots** — every spec is saved as `plot.json`; `mavpose flight.bin --spec plot.json` redraws it on any log, no model or API key needed
- **Code mode when you need it** — `--code` has the model write a pandas + matplotlib script instead, for plots a spec can't describe
- **Restricted execution (code mode)** — generated code runs in a clean subprocess (no API keys in its environment) under a PEP 578 audit hook that blocks process spawning, network access and file writes outside the output folder, with CPU/memory limits and a timeout. This is defence in depth, not a hard security boundary: for untrusted logs or prompts, also run MAVPose in a container
- **Interactive REPL mode** — omit `--prompt` to enter a live loop; Parquet is re-extracted per query with the relevant types
- **CI-tested** — lint (ruff) and pytest run on every push

---

## Tech Stack

| Layer | Library / Service |
|---|---|
| LLM | Any provider: OpenAI, Anthropic, Gemini, OpenRouter, Groq, DeepSeek, Mistral, Together, Ollama, LM Studio, OpenAI-compatible |
| LLM framework | LangChain (LCEL) — `langchain-openai`, `langchain-anthropic` |
| Extraction layer | pandas ≥ 2.0 + pyarrow ≥ 14.0 |
| Drone log parsing | pymavlink ≥ 2.4.48 |
| Plotting | matplotlib ≥ 3.8 (in generated script) |
| Sandbox | Subprocess + PEP 578 audit hook, stripped env, resource limits, timeout |
| Config | python-dotenv |
| Lint / CI | ruff + pytest + GitHub Actions |

---

## Prerequisites

- **Python 3.10+**
- For the chat assistant only: an API key for any [supported AI provider](#choosing-an-ai-provider), or a local model server (Ollama, LM Studio)

---

## Installation

MAVPose has two layers. Install only what you need:

| You want… | Install | Footprint |
|---|---|---|
| Clean, unit-correct telemetry in pandas / Parquet (no AI) — ArduPilot and PX4 logs | `pip install mavpose` | 10 packages |
| …plus the plain-English plot assistant (`mavpose` command) | `pip install "mavpose[chat]"` | ~60 packages |
| …or an MCP server so AI agents can read your logs (`mavpose-mcp`) | `pip install "mavpose[mcp]"` | 38 packages |

The core never imports the chat layer, so `pip install mavpose` has no LLM, vector-store or plotting dependencies.

To work on MAVPose itself:

```bash
git clone https://github.com/AyushMaria/MAVPose.git
cd MAVPose
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -e ".[dev]"       # core + chat + test tools
cp template.env .env          # chat assistant only
```

Open `.env` and fill in your key. If you installed from PyPI, there's no `template.env`: just create a file named `.env` in the folder you'll run `mavpose` from (MAVPose looks there and in parent folders), or set the variables in your shell.

```env
OPENAI_API_KEY=your-key-here          # or ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, ...
MAVPOSE_MODEL=openai:<model>          # provider:model; see "Choosing an AI provider"
```

> ⚠️ Never commit `.env` — it is already listed in `.gitignore`.

---

## Usage

The `mavpose` command is the chat assistant and needs `pip install "mavpose[chat]"`; without it, the command explains what to install.

### Single prompt

```bash
mavpose flight.tlog --prompt "Plot altitude over time"
```

### Interactive REPL

Omit `--prompt` to enter a live loop:

```bash
mavpose flight.tlog

📊 Plot request > Plot battery voltage and current
📊 Plot request > Show GPS latitude and longitude
📊 Plot request > quit
```

### All CLI flags

```
usage: mavpose [-h] [--prompt PROMPT] [--model PROVIDER:MODEL] [--code] [--spec FILE] [--retries N] [--verbose] log_file

positional arguments:
  log_file              Path to a flight log: MAVLink .tlog, ArduPilot .bin/.log, or PX4 .ulg

options:
  -p, --prompt TEXT     Plot request. Omit for interactive mode.
  -m, --model P:M       Model as provider:model, e.g. openai:<model>, ollama:llama3.1:8b
                        (default: MAVPOSE_MODEL, or the provider whose key is set)
  --code                Have the model write a Python script instead of a plot spec
  --spec FILE           Redraw a saved plot spec (plot.json) on this log; no model needed
  -r, --retries N       Attempts per request when the spec or script fails (default: 3)
  -v, --verbose         Enable debug logging
```

The output plot is saved as `plot.png`, its spec as `plot.json`, and intermediate telemetry as `telemetry.parquet`, all in the same directory as your log file. Each run replaces the previous files. `mavpose` only reports success when a new plot was saved; with `--prompt` it exits with status 1 otherwise.

To redraw a plot later, or on another flight's log, without any model:

```bash
mavpose other_flight.tlog --spec plot.json
```

> **Need a sample log file?**  
> [Download here](https://drive.google.com/file/d/1BKv-NbSvYQz9XqqmyOyOhe3o4PAFDyZa/view?usp=sharing)

---

## Output Files

| File | Description |
|---|---|
| `telemetry.parquet` | Extracted, time-aligned telemetry for the relevant message types |
| `plot.png` | The plot |
| `plot.json` | The plot spec: redraw it with `--spec` |
| `plot.py` | The generated script (`--code` mode only) |

---

## Choosing an AI provider

The plot assistant works with any of these. Set the provider's key, then name the model as `provider:model` with `--model` or `MAVPOSE_MODEL`:

| Provider | Key variable | Example |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `--model openai:<model>` |
| Anthropic | `ANTHROPIC_API_KEY` | `--model anthropic:<model>` |
| Google Gemini | `GEMINI_API_KEY` | `--model gemini:<model>` |
| OpenRouter | `OPENROUTER_API_KEY` | `--model openrouter:z-ai/glm-5.1` (model optional) |
| Groq / DeepSeek / Mistral / Together | `GROQ_API_KEY` / `DEEPSEEK_API_KEY` / `MISTRAL_API_KEY` / `TOGETHER_API_KEY` | `--model groq:<model>` |
| Ollama / LM Studio (local) | none | `--model ollama:llama3.1:8b` |
| Any OpenAI-compatible server | `MAVPOSE_API_KEY` (if needed) | `MAVPOSE_BASE_URL=http://host:8000/v1` and `--model openai-compatible:<model>` |

Existing setups with only `OPENROUTER_API_KEY` (and optionally `OPENROUTER_MODEL`) keep working as before. If exactly one provider key is set, a bare `--model <model>` is enough. Details: [Plot assistant guide](docs/guide/chat.md#choosing-an-ai-provider).

---

## Project Structure

```
MAVPose/
├── mavpose/                       # pip install mavpose — core data layer
│   ├── flightlog.py               # mavpose.load() and FlightLog: the public API
│   ├── mcp_server.py              # mavpose-mcp: MCP tools for AI agents ([mcp] extra)
│   ├── log_extractor.py           # LogExtractor: one clock, real units, unknown markers
│   ├── ulog_reader.py             # PX4 ULog (.ulg) support via pyulog
│   ├── file_validator.py          # File validation (extension, size, symlink)
│   ├── data/
│   │   ├── mavlink_unknown_markers.json   # generated from the MAVLink spec
│   │   └── px4_fields.json                # units + markers from PX4 msg definitions
│   └── chat/                      # pip install "mavpose[chat]" — plot assistant
│       ├── plot_creator.py        # PlotCreator: LLM writes the plot script
│       ├── llm.py                 # Choosing the AI provider and model
│       ├── safe_executor.py       # Restricted subprocess executor (audit hook + limits)
│       └── cli.py                 # `mavpose` command
├── docs/                          # guide, format notes, API reference (MkDocs)
├── CHANGELOG.md
├── tools/
│   ├── gen_unknown_markers.py     # regenerates MAVLink markers from mavlink/mavlink
│   └── gen_px4_field_info.py      # regenerates PX4 units/markers from PX4-Autopilot/msg
├── tests/
│   ├── log_builders.py            # builds real .tlog / .bin files for tests
│   ├── test_real_logs.py          # end-to-end extraction tests
│   ├── test_core_boundary.py      # core must not depend on chat
│   └── …
├── .github/workflows/ci.yml       # ruff + pytest on 3.10–3.13, core-only job, strict docs build
├── template.env
└── LICENSE
```

---

## Library API

```python
import mavpose

log = mavpose.load("flight.tlog")          # .tlog, .bin, .log or .ulg
log
# <FlightLog 'flight.tlog': mavlink, 38 message types, 2443.8 s>

alt = log.series("GLOBAL_POSITION_INT", "relative_alt")   # Series indexed by time_s
alt.attrs["unit"]                                         # 'm'

log.unknown_counts        # {'GPS_RAW_INT': {'epv': 6338}}  "not provided" values now NaN
log.to_parquet("telemetry.parquet")
```

`mavpose.load`, `FlightLog`, `LogExtractor`, `validate_mavlink_file` and
`FileValidationError` are the public API, covered by the
[stability policy](docs/api-stability.md). Full documentation:

- [Concepts](docs/guide/concepts.md): time axis, units, unknown values, damaged logs
- [Log formats](docs/guide/formats.md): message names and quirks per format
- [API reference](docs/api/flightlog.md) and [changelog](CHANGELOG.md)

To preview the docs locally: `pip install -e ".[docs]" && mkdocs serve`.

## MCP server for AI agents

```bash
pip install "mavpose[mcp]"
mavpose-mcp --root ~/flight-logs
```

Any MCP-capable assistant or agent framework can then call `list_logs`,
`summarize_flight`, `list_messages`, `describe_message`, `get_series`,
`get_log_messages` and `export_parquet`. The server is limited to the
`--root` folders, read-only except export, and never runs code. Long series
are downsampled for the agent's context, but min/max/mean always cover every
sample. See [docs/guide/mcp.md](docs/guide/mcp.md) for client setup.

## Plot assistant API — `PlotCreator`

```python
from mavpose.chat import PlotCreator   # needs mavpose[chat]

creator = PlotCreator(max_retries=3, model="openai:<model>")   # or None: MAVPOSE_MODEL / detected key
creator.set_logfile_name("flight.tlog")

# Phase 1a: schema scan
creator.parse_mavlink_log()

# Phase 1b: the model picks message types; Parquet extraction
msg_types = creator.find_relevant_data_types("Plot altitude over time")
schema_summary = creator.extract_dataframes(msg_types)

# Phase 2: the model designs the plot as a JSON spec; MAVPose draws it
ok = creator.plot_from_spec("Plot altitude over time", schema_summary)
# → plot.png, plot.json and telemetry.parquet saved next to the log file
# (PlotCreator(mode="code") + create_plot() / run_script() for code mode)
```

---

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

To lint:
```bash
ruff check .
```

In a core-only install (`pip install -e .`), the chat tests skip themselves.

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `No AI provider is configured` / `..._API_KEY is not set` | No key in the shell or in a `.env` file | Set your provider's key (e.g. `OPENAI_API_KEY`) in `.env` in the folder you run `mavpose` from, or export it |
| `Choose which ... model to use` | Key found, but no model named | Add `--model provider:<model>` or `MAVPOSE_MODEL=provider:<model>` |
| `ModuleNotFoundError: pandas` | Missing dependency | Run `pip install mavpose` |
| `ModuleNotFoundError: pyarrow` | Missing dependency | Run `pip install mavpose` |
| "The MAVPose chat assistant needs extra packages" | Core-only install | Run `pip install "mavpose[chat]"` |
| Plot not generated after N retries | LLM script failed repeatedly | Try a more specific prompt; use `--verbose` to inspect errors |
| `FileValidationError` | Wrong file type or empty file | Only `.tlog`, `.bin`, `.log` and `.ulg` files ≤ 200 MB are accepted |
| `Couldn't tell which message types this request needs` | The model didn't name any message type in the log | Name the data more specifically; use `--verbose` to see the model's answer |

---

## Contributing

Contributions are welcome, from bug reports about logs MAVPose reads wrongly to code. See [CONTRIBUTING.md](CONTRIBUTING.md) to get set up, and the [`good first issue`](https://github.com/AyushMaria/MAVPose/labels/good%20first%20issue) label for a place to start. Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).

---

## License

MIT — see [LICENSE](LICENSE) for details.
