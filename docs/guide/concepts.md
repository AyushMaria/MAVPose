# Concepts

Four guarantees apply to every format MAVPose reads. This page explains
each one once.

## One time axis

Every frame has a `time_s` column: **seconds from the start of the log,
on one clock shared by all message types**. A battery reading at
`time_s = 120.5` and an attitude reading at `time_s = 120.5` happened at
the same moment.

Where the clock comes from depends on the format:

| Format | Source of `time_s` | Start (`time_s = 0`) |
|---|---|---|
| `.tlog` | the ground station's receive timestamp, stored with every packet | first message |
| `.bin` / `.log` | the vehicle's `TimeUS`, anchored to GPS time when GPS is available | first message |
| `.ulg` | each topic's `timestamp` (µs since boot) | the logging-start time in the file header |

The log's own time fields are kept as ordinary columns too
(`time_boot_ms`, `time_usec`, `TimeUS`, `timestamp`), so nothing is lost.

!!! note "Why not use the time fields inside each message?"
    In MAVLink logs those fields use different clocks. `SYSTEM_TIME`
    counts from the Unix epoch, `GPS_RAW_INT` often counts from boot, and
    many messages have no time field at all. Mixing them puts messages
    billions of seconds apart.

**ULog specifics.** A few PX4 topics hold a value published shortly *before*
logging began; those rows keep small negative times (for example
`-0.2`). Rows whose timestamp is `0` were never stamped by PX4, so they
are dropped and a warning reports how many.

## Units

Raw logs store many values as scaled integers. MAVPose converts them to
natural units, so `GLOBAL_POSITION_INT.alt` reads `608.81` (m) instead of
`608810` (mm).

| Raw unit | Converted to | Example field |
|---|---|---|
| `mm`, `cm` | `m` | `GLOBAL_POSITION_INT.alt` |
| `degE7` | `deg` | `GLOBAL_POSITION_INT.lat` |
| `cdeg` | `deg` | `GLOBAL_POSITION_INT.hdg` |
| `cm/s`, `mm/s` | `m/s` | `GLOBAL_POSITION_INT.vx` |
| `mV`, `cV` | `V` | `SYS_STATUS.voltage_battery` |
| `cA`, `mA` | `A` | `SYS_STATUS.current_battery` |
| `d%`, `c%` | `%` | `SYS_STATUS.load` |

Time units (`ms`, `us`) are never rescaled.

**Where units come from**, so you can judge how far to trust them:

- **MAVLink:** pymavlink's unit metadata, generated from the official
  MAVLink message definitions. It covers every message.
- **DataFlash:** the log's own `FMTU`/`UNIT`/`MULT` records, which modern
  ArduPilot writes. Older logs without them still get correct values but
  have no unit labels.
- **ULog:** in order of preference,
    1. PX4's message annotations (e.g. `float32 current_a # [A]`),
    2. a small table for older unannotated fields,
    3. PX4's naming convention (`_m_s2`, `_rad`, `_v`, `_a`, …).

    Old PX4 logs store GPS position as integers (degE7, mm); these are
    converted to match newer logs.

Read the unit of any column with `log.unit(msg, field)`, from
`series.attrs["unit"]`, or for everything at once from `log.units`.
`None` means the unit is unknown or the field is dimensionless.

Pass `convert_units=False` to keep the raw stored values, with their
raw units reported:

```python
raw = mavpose.load("flight.tlog", convert_units=False)
raw.unit("GLOBAL_POSITION_INT", "alt")     # 'mm'
```

## Unknown values

Autopilots send sentinel values when a reading isn't available: battery
current `-1`, GPS accuracy `65535`, an unused battery-cell slot at `0 V`.
Plotted as-is, these show up as fake dips and spikes. MAVPose replaces them
with NaN.

The sentinels are not guessed. They come from the `invalid` markers in
the official MAVLink definitions (441 fields) and PX4's message
annotations (31 fields), and ship with MAVPose as generated tables. A
handful of fields where the spec's marker is also a normal reading (for
example, a 0 KiB/s data rate) are deliberately excluded.

You can see how many values were replaced:

```python
log.unknown_counts
# {'GPS_RAW_INT': {'epv': 6338}}      # this GPS never reports vertical accuracy
# In a log where the current sensor needed 3 samples to start up:
log.summary(["SYS_STATUS"])["SYS_STATUS"]["columns"]["current_battery"]
# {'dtype': 'float64', 'unit': 'A', 'min': 10.5, 'max': 10.5, 'unknown': 3}
```

Pass `filter_unknown=False` to keep the raw markers. DataFlash logs have no
such markers and are never filtered.

## Damaged logs

Real logs get truncated by power loss or corrupted in transfer. MAVPose
skips bad packets instead of stopping, and keeps everything it can read:

```python
log.stats
# {'messages': 163851, 'bad_data': 0, 'errors': 0, 'unknown_values': 6338}
```

- `bad_data` counts corrupt packets that were skipped.
- `errors` counts reader exceptions that were recovered from.

A warning is logged whenever either is non-zero. Parsing only gives up
after 1,000 consecutive read errors, which means the rest of the file is
unreadable.

## Exporting

`log.to_parquet(path)` writes all message types (or a chosen list) to one
Parquet file, with a `msg_type` column to tell them apart. `log.summary()`
returns, for each column, its dtype, unit, min/max and unknown count. This
is the schema MAVPose's plot assistant gives to the language model, and it's
handy for your own tooling too.
