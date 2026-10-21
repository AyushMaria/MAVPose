# Contributing to MAVPose

Thanks for helping. MAVPose exists so that drone developers don't each have
to rediscover how to read flight logs correctly, and it gets better with
every log format quirk, unit fix and bug report from people who fly real
vehicles.

You don't need to write code to help:

- **Report a log MAVPose reads wrongly.** A wrong unit, a fake reading that
  wasn't filtered, a time axis that doesn't line up: these are the most
  valuable bug reports we get.
- **Improve the docs**, for example by explaining a message type or a format quirk you ran into.
- **Answer questions** in issues.

Everyone taking part is expected to follow the
[code of conduct](CODE_OF_CONDUCT.md).

## Finding something to work on

- Issues labelled
  [`good first issue`](https://github.com/AyushMaria/MAVPose/labels/good%20first%20issue)
  are small, self-contained and have a suggested approach.
- Issues labelled
  [`help wanted`](https://github.com/AyushMaria/MAVPose/labels/help%20wanted)
  are bigger, and we'd welcome help with them.
- For anything larger, or a change to how the data looks (time, units,
  unknown values, message names), please open an issue first so we can agree
  on the approach before you spend time on it.

Comment on an issue to say you're working on it, so two people don't do the
same work.

## Setting up

You need Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/MAVPose.git
cd MAVPose
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + chat + MCP + test tools
```

You don't need an OpenRouter API key to work on MAVPose: the tests never call
a language model.

## Checks to run before opening a pull request

These are the same checks CI runs:

```bash
python -m pytest                    # all tests
ruff check .                        # lint
pip install -e ".[docs]" && mkdocs build --strict   # only if you changed docs/
```

CI also runs the tests on Python 3.10–3.13 and in a core-only install
(`pip install -e .`), where the chat and MCP tests skip themselves.

To run the lint checks automatically before every commit, install the
pre-commit hooks once:

```bash
pre-commit install
```

They run `ruff check` and fix trailing whitespace and missing final
newlines. Run them on everything with `pre-commit run --all-files`.

## How the code is organised

| Path | What it is |
|---|---|
| `mavpose/flightlog.py` | `mavpose.load()` and `FlightLog`, the main public API |
| `mavpose/log_extractor.py` | Parsing `.tlog` / `.bin` / `.log`, time axis, units, unknown markers |
| `mavpose/ulog_reader.py` | PX4 `.ulg` support |
| `mavpose/file_validator.py` | Input checks |
| `mavpose/data/` | Generated unit and unknown-marker tables (don't edit by hand) |
| `mavpose/mcp_server.py` | The `mavpose-mcp` server (`[mcp]` extra) |
| `mavpose/chat/` | The plot assistant and its script runner (`[chat]` extra) |
| `tools/` | Scripts that regenerate `mavpose/data/` from MAVLink and PX4 definitions |
| `tests/log_builders.py` | Builds small, realistic logs of each format for tests |
| `docs/` | The documentation site (MkDocs) |

Three rules keep the project healthy:

1. **The core never imports the optional layers.** `pip install mavpose` must
   stay small and free of AI dependencies. Code in `mavpose/` outside `chat/`
   and `mcp_server.py` must not import from them, and a test enforces this.
2. **The data guarantees are public API.** One shared `time_s` axis, unit
   conversion, unknown-value filtering and message naming are covered by the
   [stability policy](docs/api-stability.md). A change to any of them is a
   breaking change and needs an issue first.
3. **Generated data is generated.** To update `mavpose/data/`, change and
   rerun the script in `tools/` (each script's docstring says how), and
   commit the script change and the regenerated file together.

## Tests

- Every bug fix needs a test that fails without the fix.
- Prefer building a log with `tests/log_builders.py` over adding binary
  files. If a real log is the only way to reproduce a problem, keep it small,
  make sure you're allowed to share it, and strip anything private first:
  **flight logs contain GPS positions**, often including where you live or
  work.
- Check numbers against known values: a test that only checks "no error"
  doesn't prove a unit or a time axis is right.

## Pull requests

- Keep each pull request to one change, and link the issue it fixes
  (`Fixes #123`).
- Add a line under the next version's heading in `CHANGELOG.md` for anything
  users will notice. If there's no heading for the next version yet, add one
  marked "unreleased".
- Update the docs if behaviour or options change.
- CI must pass before a pull request is merged.

## Reporting bugs and security issues

Use the [issue templates](https://github.com/AyushMaria/MAVPose/issues/new/choose).
For a log-reading bug, say which autopilot and firmware version wrote the
log, and attach the log if you can share it (see the GPS warning above).

If you find a security problem, for example a way for a generated plot
script to escape the plot assistant's restrictions, please don't open a
public issue. Email **maria.ayush@gmail.com** instead.

## Releases

Maintainers release by publishing a GitHub Release tagged `vX.Y.Z` that
matches the version in `pyproject.toml`. A workflow tests the package and
uploads it to PyPI.
