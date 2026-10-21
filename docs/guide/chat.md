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
it. Small models work best with specific requests ("plot battery voltage
over time"); they're less reliable at writing correct plotting code than
large hosted models.

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
3. The model writes a short pandas + matplotlib script, which MAVPose runs
   and, if it fails, asks the model to fix (up to `--retries` times).

## Safety

The generated script runs in a separate process:

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
