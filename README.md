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

MAVPose is a headless CLI tool that turns a natural language prompt into a matplotlib plot of your MAVLink flight log. It runs a **two-phase pipeline**: the parent process first extracts clean, time-aligned telemetry into a Parquet file (like a headless database query), then hands the LLM a precise column schema and a `pd.read_parquet()` call — no raw binary data, no pymavlink in the generated script. This drastically reduces hallucinations and self-healing loops.

```
$ python cli.py flight.tlog --prompt "Plot altitude over time"

📂 Parsing log schema: flight.tlog
✅ Schema indexed.

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

✍️  Generating plot script with z-ai/glm-5.1...
⚙️  Running script...
✅ Plot saved to: /path/to/plot.png
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
│    └─ ChromaDB embeddings (semantic field search)           │
│                                                             │
│  find_relevant_data_types(prompt)                           │
│    └─ vector similarity → ["GLOBAL_POSITION_INT", ...]      │
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
│    └─ parquet_file path                                    │
│    └─ schema: {msg_type: {rows, columns: {col: dtype+range}}} │
│    └─ output_file path (.png)                              │
│                                                             │
│  LLM writes pandas + matplotlib script                      │
│    df = pd.read_parquet(parquet_file)                       │
│    df_x = df[df['msg_type'] == 'X']                         │
│    ...                                                      │
│                                                             │
│  run_script() → subprocess exec                            │
│    └─ on failure: attempt_to_fix_script() up to N×         │
│                                                             │
│  → plot.png saved next to log file                          │
└──────────────────────────────────────────────────────────┘
```

**Why this matters:** In the old design, the LLM had to write `pymavlink` code to parse a binary `.tlog` — a notoriously tricky API with blocking/non-blocking subtleties, timestamp inconsistencies, and message-type guesswork. By the time the LLM touched the data, it was flying blind. Now it receives a typed, time-aligned Parquet file with exact column names, dtypes, and value ranges. Writing `pd.read_parquet()` + `matplotlib` code against a known schema is trivially reliable.

---

## Features

- **Plain-English plots** — describe any flight data; MAVPose figures out which MAVLink fields to use
- **Headless extraction layer** — `LogExtractor` parses MAVLink `.tlog` and ArduPilot DataFlash `.bin`/`.log` files into per-message-type DataFrames before the LLM is ever invoked:
  - **one clock** — every message type shares a single `time_s` axis (tlog receive time, or DataFlash `TimeUS`); native time fields are kept as columns
  - **real units** — scaled fields are converted using the log's own unit metadata (`alt` mm → m, `lat` degE7 → deg, `voltage_battery` mV → V, …) and every column's unit is reported; pass `convert_units=False` for raw values
  - **corruption-tolerant** — bad packets are skipped and counted (`extractor.stats`) instead of silently ending the parse
- **Clean Parquet handoff** — only the relevant message types are exported; the LLM sees exact column names, dtypes, min/max ranges — no binary guesswork
- **Semantic field search** — ChromaDB vector embeddings surface the most relevant message types for your query
- **Self-healing scripts** — LLM debugs and rewrites failing scripts up to N times; fix prompt also includes the full schema
- **Restricted execution** — generated code runs in a clean subprocess (no API keys in its environment) under a PEP 578 audit hook that blocks process spawning, network access and file writes outside the output folder, with CPU/memory limits and a timeout. This is defence in depth, not a hard security boundary: for untrusted logs or prompts, also run MAVPose in a container
- **Interactive REPL mode** — omit `--prompt` to enter a live loop; Parquet is re-extracted per query with the relevant types
- **Configurable model** — drop in any OpenRouter model with a one-line `.env` change
- **Persisted vector store** — ChromaDB is saved to disk; re-running on the same log skips re-embedding
- **CI-tested** — lint (ruff) and pytest run on every push

---

## Tech Stack

| Layer | Library / Service |
|---|---|
| LLM | [OpenRouter](https://openrouter.ai) → Z.ai GLM-5.1 (default) |
| Embeddings | OpenRouter → `openai/text-embedding-3-small` |
| LLM framework | LangChain (LCEL) — `langchain-openai`, `langchain-chroma` |
| Vector store | ChromaDB ≥ 0.5 (persisted to disk) |
| Extraction layer | pandas ≥ 2.0 + pyarrow ≥ 14.0 |
| Drone log parsing | pymavlink 2.4.37 |
| Plotting | matplotlib 3.7.1 (in generated script) |
| Sandbox | Subprocess + PEP 578 audit hook, stripped env, resource limits, timeout |
| Config | python-dotenv |
| Lint / CI | ruff + pytest + GitHub Actions |

---

## Prerequisites

- **Python 3.10+**
- An **OpenRouter API key** — [get one here](https://openrouter.ai/keys) (free tier available)

---

## Installation

```bash
# 1. Clone
git clone https://github.com/AyushMaria/MAVPose.git
cd MAVPose

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp template.env .env
```

Open `.env` and fill in your key:

```env
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=z-ai/glm-5.1   # default — change to any OpenRouter model
```

> ⚠️ Never commit `.env` — it is already listed in `.gitignore`.

---

## Usage

### Single prompt

```bash
python cli.py flight.tlog --prompt "Plot altitude over time"
```

### Interactive REPL

Omit `--prompt` to enter a live loop:

```bash
python cli.py flight.tlog

📊 Plot request > Plot battery voltage and current
📊 Plot request > Show GPS latitude and longitude
📊 Plot request > quit
```

### All CLI flags

```
usage: mavpose [-h] [--prompt PROMPT] [--retries N] [--verbose] log_file

positional arguments:
  log_file              Path to a MAVLink log file (.tlog, .bin, .log)

options:
  -p, --prompt TEXT     Plot request. Omit for interactive mode.
  -r, --retries N       Max self-healing retries on script failure (default: 3)
  -v, --verbose         Enable debug logging
```

The output plot is saved as `plot.png`, and intermediate telemetry as `telemetry.parquet`, both in the same directory as your log file.

> **Need a sample log file?**  
> [Download here](https://drive.google.com/file/d/1BKv-NbSvYQz9XqqmyOyOhe3o4PAFDyZa/view?usp=sharing)

---

## Output Files

| File | Description |
|---|---|
| `telemetry.parquet` | Extracted, time-aligned telemetry for the relevant message types |
| `plot.py` | The LLM-generated pandas + matplotlib script |
| `plot.png` | The final plot at 400 dpi |
| `chroma_db/` | Persisted ChromaDB vector store (git-ignored) |

---

## Switching Models

MAVPose routes through [OpenRouter](https://openrouter.ai/models), so any model on the platform works. Just update `.env`:

```env
# Z.ai options
OPENROUTER_MODEL=z-ai/glm-5.1          # default — long-horizon agentic coding
OPENROUTER_MODEL=z-ai/glm-5            # flagship, complex systems
OPENROUTER_MODEL=z-ai/glm-5-turbo      # fast inference
OPENROUTER_MODEL=z-ai/glm-4.5          # MoE, 355B params, switchable reasoning

# OpenAI fallback
OPENROUTER_MODEL=openai/gpt-4o
```

---

## Project Structure

```
MAVPose/
├── cli.py                         # CLI entry point — two-phase orchestration
├── app.py                         # Legacy stub (Gradio UI removed)
├── llm/
│   ├── log_extractor.py           # 🆕 Headless extraction layer (LogExtractor)
│   ├── gptPlotCreator.py          # PlotCreator — orchestrates both phases
│   ├── safe_executor.py           # Restricted subprocess executor (audit hook + limits)
│   └── file_validator.py          # File validation (extension, size, symlink)
├── tests/
│   ├── test_log_extractor.py      # 🆕 Unit tests for LogExtractor
│   ├── test_extract_code_snippets.py
│   ├── test_file_validator.py
│   └── test_safe_executor.py
├── .github/workflows/ci.yml       # GitHub Actions: ruff lint + pytest
├── docs/
│   └── GPT_MAVPlot_Arch.png
├── target/                        # Output directory for plot.py and plot.png
├── template.env
├── requirements.txt
└── LICENSE
```

---

## Core API — `LogExtractor`

```python
from llm.log_extractor import LogExtractor

extractor = LogExtractor("flight.tlog")

# Fast schema scan (no DataFrame allocation)
schema = extractor.schema_only()
# {"GLOBAL_POSITION_INT": {"count": 1842, "fields": {"lat": "int", ...}, "units": {"lat": "deg", ...}}, ...}

# Full extraction into DataFrames
frames = extractor.extract_all()
# {"GLOBAL_POSITION_INT": pd.DataFrame([time_s, msg_type, lat, lon, alt, ...]), ...}
extractor.units["GLOBAL_POSITION_INT"]["alt"]   # "m"
extractor.stats                                   # {"messages": ..., "bad_data": 0, "errors": 0}

# Export to Parquet and get schema summary for the LLM
summary = extractor.export_parquet(["GLOBAL_POSITION_INT", "VFR_HUD"], "telemetry.parquet")
# {"GLOBAL_POSITION_INT": {"rows": 1842, "columns": {"alt": {"dtype": "float64", "unit": "m", "min": 487.32, "max": 512.1}, ...}}}
```

## Core API — `PlotCreator`

```python
from mavpose import PlotCreator

creator = PlotCreator(max_retries=3)
creator.set_logfile_name("flight.tlog")

# Phase 1a: schema scan + embeddings
creator.parse_mavlink_log()

# Phase 1b: semantic search + Parquet extraction
msg_types = creator.find_relevant_data_types("Plot altitude over time")
schema_summary = creator.extract_dataframes(msg_types)

# Phase 2: LLM writes + executes the script
creator.create_plot("Plot altitude over time", schema_summary)
result, code = creator.run_script()
# → plot.png and telemetry.parquet saved next to the log file
```

---

## Running Tests

```bash
pip install pytest pytest-cov ruff
pytest tests/ -v --cov=llm
```

To lint:
```bash
ruff check llm/ cli.py
```

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `KeyError: OPENROUTER_API_KEY` | `.env` not configured | Copy `template.env` to `.env` and add your key |
| `ModuleNotFoundError: pandas` | Missing dependency | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: pyarrow` | Missing dependency | Run `pip install -r requirements.txt` |
| Plot not generated after N retries | LLM script failed repeatedly | Try a more specific prompt; use `--verbose` to inspect errors |
| `FileValidationError` | Wrong file type or empty file | Only `.tlog`, `.bin`, `.log` files ≤ 200 MB are accepted |
| `ValueError: None of the requested message types were found` | Semantic search returned types not in log | Use `--verbose` to see what types the log actually contains |
| ChromaDB version conflict | Stale venv | Delete `venv/` and reinstall with a fresh `pip install -r requirements.txt` |

---

## License

MIT — see [LICENSE](LICENSE) for details.
