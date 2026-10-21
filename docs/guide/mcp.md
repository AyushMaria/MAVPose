# MCP server (for AI agents)

`mavpose-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io)
server. Any MCP-capable AI assistant or agent framework can use it to read
your flight logs, with MAVPose's guarantees: one time axis, real units, and
"value not provided" markers removed.

```bash
pip install "mavpose[mcp]"
mavpose-mcp --root ~/flight-logs
```

## Connect a client

MCP clients start the server themselves over stdio. Most use a JSON entry
like this one; the exact file and key names depend on the client:

```json
{
  "mcpServers": {
    "mavpose": {
      "command": "mavpose-mcp",
      "args": ["--root", "/home/you/flight-logs"]
    }
  }
}
```

If `mavpose-mcp` isn't on the client's `PATH` (for example, it's installed
in a virtual environment), give the full path to the command, or use
`"command": "/path/to/python", "args": ["-m", "mavpose.mcp_server", "--root", "..."]`.

## Tools

| Tool | What it returns | Writes? |
|---|---|---|
| `list_logs` | Log files (`.tlog`, `.bin`, `.log`, `.ulg`) in the allowed folders, with format and size | no |
| `summarize_flight` | Duration; altitude, speed, battery and GPS ranges (each with its source field and unit); parse stats; unknown-value counts; warnings the vehicle reported | no |
| `list_messages` | Message types with row counts, field counts and time range | no |
| `describe_message` | One message type's fields: dtype, unit, min, max, unknown count | no |
| `get_series` | Fields of one message type over a time window, on the shared `time_s` axis, with units | no |
| `get_log_messages` | The vehicle's text log (MAVLink `STATUSTEXT`, ArduPilot `MSG`, PX4 log messages), filterable by severity | no |
| `export_parquet` | Writes message types to a `.parquet` file inside an allowed folder | **yes**, never overwrites unless `overwrite` is true |

A typical agent session: `list_logs` → `summarize_flight` → `list_messages`
→ `describe_message` → `get_series`, zooming in with `start_s`/`end_s`, and
`get_log_messages` to see what the vehicle reported.

### Built for an agent's context window

`get_series` returns at most `max_points` samples (500 by default, up to
5,000). Long windows are downsampled to evenly spaced points, **but `min`,
`max` and `mean` are always computed over every sample in the window**, so
an agent can't miss a brief spike because it fell between the returned
points. For example, on a 5-minute window of a real flight, `get_series`
with `max_points=5` returned currents of 0.63–0.78 A, and still reported
the window's true peak of 28.97 A.

Unknown values are `null` in results, and each field reports its
`null_count`.

### Helpful errors

Mistakes come back as plain messages the agent can act on:

```text
No message type 'GLOBAL_POS'. Did you mean: GLOBAL_POSITION_INT, LOCAL_POSITION_NED?
```

## Safety

- **Folder allowlist.** The server only reads, and `export_parquet` only
  writes, inside the `--root` folders. Pass `--root` several times for several
  folders. If you give none, it uses `$MAVPOSE_MCP_ROOTS` (folders separated by
  `:` on Linux and macOS, `;` on Windows), and otherwise the current directory.
  Paths that resolve outside these folders, including via `..`, are refused.
- **No symlinked logs,** and only log file extensions are read.
- **No code execution.** Every tool calls MAVPose's library directly. The
  server doesn't use the plot assistant's script runner.
- **Read-only annotations.** Every tool except `export_parquet` is marked
  read-only and non-destructive, so clients can approve them more freely.

## Notes

- **Caching.** Parsed logs are cached (four at a time) and re-read
  automatically if the file changes, so repeated calls on the same log are fast.
- **Text-message severities.** Some older ArduPilot firmware used
  non-standard `STATUSTEXT` severity numbers, so severity levels from old
  `.tlog` files can be unreliable. Results include the raw `severity` number
  so an agent can judge.
- **Transport.** The server speaks MCP over stdio, the way local clients
  launch servers.
