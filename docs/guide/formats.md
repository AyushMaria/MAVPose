# Log formats

`mavpose.load()` picks the reader from the file extension. Everything
described in [Concepts](concepts.md) applies to all formats; this page covers
what differs between them.

## MAVLink telemetry logs (`.tlog`)

Recorded by a ground station (Mission Planner, QGroundControl, MAVProxy)
from the telemetry link, so they contain what the vehicle *sent*, at the
telemetry rate.

- **Message names** are MAVLink message names: `GLOBAL_POSITION_INT`,
  `ATTITUDE`, `SYS_STATUS`, `VFR_HUD`, `GPS_RAW_INT`, …
- **Field names** match the
  [MAVLink message definitions](https://mavlink.io/en/messages/common.html).
- **Time** comes from the receive timestamp stored before every packet.
- **Units and unknown markers** come from the MAVLink definitions, and
  cover every message.

## ArduPilot DataFlash logs (`.bin`, `.log`)

Written on the vehicle's SD card, at full sensor rates. `.bin` is binary;
`.log` is the text form of the same data.

- **Message names** are ArduPilot log message names: `ATT`, `BARO`,
  `BAT`, `GPS`, `IMU`, `CTUN`, …  See the
  [ArduPilot log message reference](https://ardupilot.org/copter/docs/logmessages.html).
- **Time** comes from each message's `TimeUS`, anchored to GPS time when
  the log has a GPS fix.
- **Units** come from the log's `FMTU`/`UNIT`/`MULT` records, which recent
  ArduPilot versions write. Older logs without them load with correct
  values but no unit labels. Fields stored with a fixed scale
  (centidegrees, degE7) are already scaled by pymavlink and are never
  scaled twice.
- The `FMT`, `FMTU`, `UNIT` and `MULT` records describe the log itself
  and are not returned as data.

## PX4 ULog (`.ulg`)

Written by PX4's logger, one table per uORB topic.

- **Message names** are topic names: `vehicle_attitude`,
  `battery_status`, `sensor_combined`, `vehicle_gps_position`, …
- **Multiple instances** of a topic become separate message types:
  `battery_status` (instance 0), `battery_status_1`, `battery_status_2`, …
- **Array fields** are flattened with pyulog's names: `q[0]`…`q[3]`,
  `voltage_cell_v[0]`, …
- **PX4's text log** (`INFO [commander] Takeoff detected`) becomes a
  `logged_messages` message type with `level` and `message` columns.
- **Time** comes from each topic's `timestamp`, measured from the logging
  start (see [ULog time](concepts.md#one-time-axis)).
- **Units** come from PX4's message annotations, then a fallback table, then
  PX4's naming convention (see [Units](concepts.md#units)).

```python
log = mavpose.load("flight.ulg")
log.series("battery_status", "voltage_v").attrs["unit"]    # 'V'
log.frame("logged_messages")[["time_s", "level", "message"]]
```

## Which file should I use?

If you have it, use the **on-board log** (`.bin` for ArduPilot, `.ulg` for
PX4). It has every sensor at full rate. A `.tlog` only holds what was
streamed over telemetry, usually at a few Hz, but it's often all you have
from a ground station.
