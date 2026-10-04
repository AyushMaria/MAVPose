# Plot assistant (optional)

`mavpose[chat]` adds a command that turns a plain-English request into a
plot:

```bash
pip install "mavpose[chat]"
cp template.env .env          # add your OPENROUTER_API_KEY
mavpose flight.tlog --prompt "Plot altitude and battery voltage"
mavpose flight.ulg            # interactive mode: ask several questions
```

## How it works

1. MAVPose reads the log with the core library (all of [Concepts](concepts.md)
   applies) and finds the message types relevant to your question.
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
