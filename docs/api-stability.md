# API stability

MAVPose follows [Semantic Versioning](https://semver.org/). This page
says exactly what that promise covers, so other projects can build on
MAVPose with confidence.

## Public API

These names are public. Their documented behaviour will not change in a
backwards-incompatible way without the process below.

| Name | What it is |
|---|---|
| `mavpose.load()` | Read a log into a `FlightLog` |
| `mavpose.FlightLog` | `messages()`, `fields()`, `frame()` / `log[...]`, `series()`, `unit()`, `units`, `unknown_counts`, `stats`, `duration_s`, `summary()`, `to_parquet()`, `format`, `path` |
| `mavpose.LogExtractor` | Lower-level extractor: `schema_only()`, `extract_all()`, `summary()`, `export_parquet()`, `frames`, `units`, `raw_units`, `unknown_counts`, `stats` |
| `mavpose.validate_mavlink_file()`, `mavpose.FileValidationError` | Input validation |
| `mavpose.__version__` | Installed version |
| The `mavpose` command | Plot assistant CLI (needs `[chat]`) |

The **data guarantees** are part of the public API too: one shared
`time_s` axis, unit conversion, unknown-marker filtering, and the message
naming described in [Log formats](guide/formats.md). Changing any of
these, for example a different time origin for a format, counts as a
breaking change.

**Not public:** anything whose name starts with `_`; the modules
`mavpose.ulog_reader` and `mavpose.chat.*` except `mavpose.chat.PlotCreator`;
and the files under `mavpose/data/`. These may change in any release.

## What a version number tells you

MAVPose is **pre-1.0**. Until 1.0:

- **Minor releases** (0.2 → 0.3) may make breaking changes. Each one is
  listed under "Breaking" in the [changelog](changelog.md).
- **Patch releases** (0.2.0 → 0.2.1) contain only fixes and additions,
  never breaking changes.

From 1.0, breaking changes happen only in major releases.

**Generated tables change too.** A fix to the MAVLink or PX4
unit/marker tables can change the values you see, for example a field that
was a fake `-1` becoming NaN. That's treated as a bug fix and noted in the
changelog.

## Deprecation

Before a public name is removed or changed incompatibly, it keeps working
for **at least one minor release** and emits a `DeprecationWarning` naming
its replacement. For example, the 0.1 module paths `mavpose.PlotCreator`,
`mavpose.safe_executor` and `mavpose.cli` still work in 0.2 with a warning,
and will be removed in 0.3.

To catch these early in your own test suite:

```bash
python -W error::DeprecationWarning -m pytest
```
