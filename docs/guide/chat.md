# Plot assistant (optional)

`mavpose[chat]` adds a command that turns a plain-English request into a
plot:

```bash
pip install "mavpose[chat]"
export OPENAI_API_KEY=...      # or any provider below
mavpose flight.tlog --model openai:<model> --prompt "Plot altitude and battery voltage"
mavpose flight.ulg --model openai:<model>     # interactive mode: ask several questions
```

## Choosing an AI provider

The assistant works with any of these. Set the provider's key and name a
model as `provider:model`, with `--model` or `MAVPOSE_MODEL`:

| Provider | Key variable | Example |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `--model openai:<model>` |
| Anthropic | `ANTHROPIC_API_KEY` | `--model anthropic:<model>` |
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `--model gemini:<model>` |
| OpenRouter | `OPENROUTER_API_KEY` | `--model openrouter:z-ai/glm-5.1` (model optional) |
| Groq | `GROQ_API_KEY` | `--model groq:<model>` |
| DeepSeek | `DEEPSEEK_API_KEY` | `--model deepseek:<model>` |
| Mistral | `MISTRAL_API_KEY` | `--model mistral:<model>` |
| Together | `TOGETHER_API_KEY` | `--model together:<model>` |
| Ollama (local) | none | `--model ollama:llama3.1:8b` |
| LM Studio (local) | none | `--model lmstudio:<model>` |
| Any OpenAI-compatible server | `MAVPOSE_API_KEY` (if needed) | `MAVPOSE_BASE_URL=http://host:8000/v1`, `--model openai-compatible:<model>` |

Use the model names your provider lists; MAVPose doesn't pick one for you,
because providers rename and retire models. If you don't pass `--model`:

- `MAVPOSE_MODEL` is used if set, e.g. `MAVPOSE_MODEL=anthropic:<model>`.
- Otherwise, if `OPENROUTER_API_KEY` is set, OpenRouter is used with
  `OPENROUTER_MODEL` or `z-ai/glm-5.1`, exactly as in MAVPose 0.2.
- Otherwise, if exactly one provider's key is set, that provider is used, and
  a bare model name is enough: `--model <model>`.

`MAVPOSE_BASE_URL` points any provider at a different address, for example
an Ollama server on another machine (`http://gpu-box:11434/v1`) or a proxy.
`MAVPOSE_API_KEY` overrides the key.

Settings can live in a `.env` file instead of your shell:

```env
ANTHROPIC_API_KEY=your-key-here
MAVPOSE_MODEL=anthropic:<model>
```

MAVPose reads `.env` from the folder you run `mavpose` in, or a parent
folder; a variable already set in your shell takes precedence. Never commit
`.env`.

**Local models** keep your logs' structure on your machine: nothing leaves
it. Because the assistant asks for a short plot spec rather than code, small
local models work well; specific requests ("plot battery voltage over time")
work best. Code mode (`--code`) needs a stronger model.

Each run writes `telemetry.parquet`, `plot.py` and `plot.png` next to the
log, replacing the previous ones. The command reports success only when
the script actually saved a new `plot.png`, and with `--prompt` it exits
with status 1 if no plot was created.

## How it works

1. MAVPose reads the log with the core library (all of [Concepts](concepts.md)
   applies) and shows the model the log's list of message types and field
   names, so it can pick the ones your question needs. (For logs with a very
   long list, only the message type names are sent, to fit smaller models.)
2. It exports them to Parquet and gives a language model the
   [schema summary](concepts.md#exporting): columns, units, ranges and unknown counts.
   It never sends the raw log.
3. The model describes the plot as a small JSON **plot spec**: which fields
   go in which panels, labels, colours, an optional time window. MAVPose
   checks the spec against the log and draws it with matplotlib. Axis labels
   get their units from the log. If the spec doesn't fit (a misspelt field,
   say), the exact error goes back to the model, up to `--retries` attempts.

No code from the model runs. The plot is saved as `plot.png` and its spec as
`plot.json` next to the log. Use `--out-dir DIR` to save them elsewhere (useful
when the log is on a read-only card), `--format svg` or `pdf` for reports, and
`--dpi N` to change the resolution (default 150). Ask a follow-up in interactive mode ("make the
current red", "only the first minute", "add GPS altitude too") and the model
sees your last three requests and their specs.

## Plot specs

```json
{
  "title": "Climb and battery",
  "start_s": null, "end_s": null,
  "panels": [
    {"title": "Altitude",
     "series": [{"msg_type": "GLOBAL_POSITION_INT", "field": "relative_alt"}]},
    {"title": "Battery",
     "series": [{"msg_type": "SYS_STATUS", "field": "voltage_battery"},
                {"msg_type": "SYS_STATUS", "field": "current_battery",
                 "axis": "right", "color": "tab:red"}]},
    {"kind": "scatter",
     "series": [{"msg_type": "GLOBAL_POSITION_INT", "field": "lat", "x_field": "lon"}]}
  ]
}
```

- **Panels** are stacked vertically and share the time axis; up to 6 panels
  with up to 8 series each.
- **`kind`** is `line` (default), `scatter` or `step`.
- Series are plotted against `time_s` unless **`x_field`** names another field
  of the same message type, as in the position track above.
- **`axis: "right"`** puts a series on a second y axis, for different units in
  one panel.
- **`start_s` / `end_s`** limit the plot to a time window, in seconds from the
  start of the log.

Specs are plain data, so you can edit them by hand, keep them in version
control, and redraw them on any log with the same message types, without a
model or an API key:

```bash
mavpose next_flight.bin --spec plot.json
```

From Python, use `mavpose.chat.plotspec.render_spec(spec, frames, units, "plot.png")`.

## Code mode

For a plot a spec can't describe (a histogram, a computed quantity, an FFT),
add `--code`. The model then writes a pandas + matplotlib script, saved as
`plot.py`, which MAVPose runs and, if it fails, asks the model to fix.

## Safety

Spec mode runs no generated code: the model's reply is only ever parsed as
JSON and checked against the log.

In code mode, the generated script runs in a separate process:

- with no API keys in its environment,
- under an audit hook that blocks process spawning, network access and
  file writes outside the output folder,
- with CPU, memory and time limits.

This is defence in depth, **not a security boundary**. Python cannot fully
sandbox itself. If you run the assistant on logs or prompts you don't trust,
run MAVPose inside a container as well.

## The core without the assistant

Everything in `mavpose` except `mavpose.chat` works without the `[chat]`
extra and never imports AI libraries. Without the extra, the `mavpose`
command explains how to install it.
