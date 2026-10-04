# MAVPose

**Clean, unit-correct telemetry from drone flight logs.**

MAVPose reads ArduPilot and PX4 logs into pandas DataFrames you can trust:
every message type on one time axis, values in real units, and "value not
provided" markers removed. An optional assistant turns plain-English
questions into plots.

| Format | Extension | Autopilot |
|---|---|---|
| MAVLink telemetry log | `.tlog` | ArduPilot, PX4, any MAVLink vehicle |
| DataFlash binary / text log | `.bin`, `.log` | ArduPilot |
| ULog | `.ulg` | PX4 |

## Install

```bash
pip install mavpose            # data layer only: 10 packages, no AI dependencies
pip install "mavpose[chat]"    # adds the plain-English plot assistant
```

Python 3.10 or newer.

## Quick start

```python
import mavpose

log = mavpose.load("flight.tlog")
log
# <FlightLog 'flight.tlog': mavlink, 38 message types, 2443.8 s>

log.messages()[:6]
# ['AHRS', 'AHRS2', 'ATTITUDE', 'COMMAND_ACK', 'COMMAND_LONG', 'EKF_STATUS_REPORT']

alt = log.series("GLOBAL_POSITION_INT", "relative_alt")
alt.attrs
# {'unit': 'm', 'msg_type': 'GLOBAL_POSITION_INT'}
alt.head(3)
# time_s
# 8.345965    1.93
# 8.593081    1.92
# 8.999325    1.91
# Name: relative_alt, dtype: float64

alt.plot(ylabel=f"altitude ({alt.attrs['unit']})")   # any pandas / matplotlib workflow

log.to_parquet("telemetry.parquet")                  # one tidy file for other tools
```

These outputs come from a real 40-minute ArduPilot flight log.

## What MAVPose does for you

- **One clock.** Each format has its own idea of time: receive time,
  `TimeUS` since boot, epoch microseconds, ULog timestamps. MAVPose
  puts every message type on a single `time_s` axis, so series from
  different messages line up. → [Time axis](guide/concepts.md#one-time-axis)
- **Real units.** Raw logs store altitude in millimetres, latitude in
  degE7 and voltage in millivolts. MAVPose converts these using the
  autopilot's own unit definitions and tells you the unit of every column.
  → [Units](guide/concepts.md#units)
- **No fake readings.** MAVLink and PX4 use sentinels such as
  `current_battery = -1` or `eph = 65535` to mean "not provided". These
  become NaN instead of dips and spikes in your plots.
  → [Unknown values](guide/concepts.md#unknown-values)
- **Damaged logs still load.** Corrupt packets are skipped and counted, not
  allowed to end the parse. → [Damaged logs](guide/concepts.md#damaged-logs)

## Next steps

- [Concepts](guide/concepts.md): time, units, unknown values and damaged logs, explained once.
- [Log formats](guide/formats.md): message names and quirks for each format.
- [Plot assistant](guide/chat.md): the optional plain-English plotting layer.
- [API reference](api/flightlog.md) and the [stability policy](api-stability.md).
