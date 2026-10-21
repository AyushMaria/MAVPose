#!/usr/bin/env python3
"""
mavpose/chat/cli.py

Headless CLI entry point for the MAVPose chat assistant (`mavpose` command).
Requires the optional extra:  pip install 'mavpose[chat]'.

Two-phase pipeline
------------------
  Phase 1 — Extraction
    parse_mavlink_log()   : schema scan (message types and fields)
    find_relevant_data_types() : the model picks the message types needed
    extract_dataframes()  : full extraction → clean telemetry.parquet

  Phase 2 — LLM plot generation
    create_plot()         : LLM writes pandas + matplotlib script
    run_script()          : executes script; self-heals on failure

Usage:
    mavpose <log_file> [options]

Examples:
    mavpose flight.tlog --prompt "Plot altitude over time"
    mavpose flight.tlog --prompt "Show battery voltage" --retries 5
    mavpose flight.tlog           # interactive prompt mode
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # imported lazily in main() so --help works without extras
    from mavpose.chat.plot_creator import PlotCreator


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mavpose",
        description="MAVPose — natural language drone flight log visualiser",
    )
    p.add_argument(
        "log_file",
        help="Path to a flight log: MAVLink .tlog, ArduPilot .bin/.log, or PX4 .ulg",
    )
    p.add_argument(
        "--prompt", "-p",
        default=None,
        help="Plot request (e.g. 'Plot altitude over time'). "
             "If omitted, enters interactive prompt mode.",
    )
    p.add_argument(
        "--model", "-m",
        default=None,
        metavar="PROVIDER:MODEL",
        help="Model to use, e.g. openai:<model>, anthropic:<model>, "
             "ollama:llama3.1:8b. Default: MAVPOSE_MODEL, or the provider whose "
             "API key is set (OpenRouter keeps working as before).",
    )
    p.add_argument(
        "--code",
        action="store_true",
        help="Have the model write a Python plotting script instead of a plot "
             "spec. More flexible, but runs generated code (in a restricted "
             "subprocess).",
    )
    p.add_argument(
        "--spec",
        default=None,
        metavar="FILE",
        help="Redraw a saved plot spec (e.g. plot.json) on this log. "
             "No model or API key needed.",
    )
    p.add_argument(
        "--retries", "-r",
        type=int,
        default=3,
        help="Attempts per request when the spec or script fails (default: 3)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return p


def _print_parquet_summary(schema_summary: dict) -> None:
    """Pretty-print the Parquet schema summary to stdout."""
    print("\n┌" + "─" * 58 + "┐")
    print("│  🗃️  Extracted Parquet schema" + " " * 31 + "│")
    print("├" + "─" * 58 + "┤")
    for mt, info in schema_summary.items():
        print(f"│  [{mt}]  {info['rows']} rows" + " " * max(0, 40 - len(mt) - len(str(info['rows']))) + "│")
        for col, meta in info["columns"].items():
            unit = f" {meta['unit']}" if meta.get("unit") else ""
            if "min" in meta:
                line = f"│      {col}: {meta['dtype']}{unit}  [{meta['min']} … {meta['max']}]"
            else:
                line = f"│      {col}: {meta['dtype']}{unit}"
            # Pad / truncate to fit box width
            line = line[:59].ljust(59) + "│"
            print(line)
    print("└" + "─" * 58 + "┘")


def run_once(creator: PlotCreator, prompt: str) -> bool:
    """Execute one plot request end-to-end.  Returns True if a plot was saved."""

    # --- Phase 1: choose message types ---
    print(f"\n🔍 Finding relevant message types for: '{prompt}'")
    msg_types = creator.find_relevant_data_types(prompt)
    print(f"   → {msg_types}")

    # --- Phase 1b: headless extraction → Parquet ---
    print("\n🗃️  Extracting telemetry to Parquet...")
    schema_summary = creator.extract_dataframes(msg_types)
    _print_parquet_summary(schema_summary)
    print(f"   Saved → {creator.parquet_path}")

    if creator.mode == "spec":
        print(f"\n✍️  Asking {creator.model} to design the plot...")
        creator.plot_from_spec(prompt, schema_summary)
        print("\n" + "─" * 60)
        print("Plot spec:")
        print("─" * 60)
        print(creator.last_code)
        print("─" * 60)
        if creator.last_success:
            print(f"\n✅ Plot saved to: {creator.plot_path}")
            print(f"   Spec saved to: {creator.spec_path}  (redraw with --spec)")
        else:
            print(
                f"\n❌ No plot was created: the model's spec couldn't be drawn after "
                f"{creator.max_retries} attempt(s). Run with --verbose to see each "
                "error, or try a more specific request."
            )
        return creator.last_success

    # --- Phase 2 (code mode): LLM writes the script ---
    print(f"\n✍️  Generating plot script with {creator.model}...")
    creator.create_plot(prompt, schema_summary)

    # --- Execute ---
    print("⚙️  Running script...")
    code = creator.run_script().code

    print("\n" + "─" * 60)
    print("Generated code:")
    print("─" * 60)
    print(code)
    print("─" * 60)

    if creator.last_success:
        print(f"\n✅ Plot saved to: {creator.plot_path}")
    else:
        print(
            f"\n❌ No plot was created: the script failed after {creator.max_retries} "
            "attempt(s). Check the generated code above, run with --verbose to see "
            "each error, or try a more specific request."
        )
    return creator.last_success


def interactive_mode(creator: PlotCreator) -> None:
    """Run a REPL loop accepting plot requests until the user quits."""
    print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            prompt = input("📊 Plot request > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        try:
            run_once(creator, prompt)
        except Exception as exc:
            print(f"❌ Error: {exc}")


def redraw_spec(log_file: str, spec_file: str) -> bool:
    """Draw a saved plot spec on *log_file*, without a model.  Returns success."""
    import json
    import os

    log = logging.getLogger("mavpose.cli")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Drawing plots needs matplotlib: pip install 'mavpose[chat]'", file=sys.stderr)
        return False
    from mavpose import LogExtractor, validate_mavlink_file
    from mavpose.chat.plotspec import PlotSpecError, parse_spec, render_spec

    try:
        with open(spec_file, encoding="utf-8") as fh:
            spec = parse_spec(fh.read())
        validate_mavlink_file(log_file)
    except (OSError, ValueError) as exc:   # includes PlotSpecError, FileValidationError
        log.error("%s", exc)
        return False
    out = os.path.join(os.path.dirname(os.path.abspath(log_file)), "plot.png")
    print(f"\n📂 Reading {log_file}")
    extractor = LogExtractor(log_file)
    extractor.extract_all()
    try:
        drawn = render_spec(spec, extractor.frames, extractor.units, out)
    except PlotSpecError as exc:
        log.error("The spec doesn't fit this log: %s", exc)
        return False
    print(json.dumps(drawn, indent=2))
    print(f"\n✅ Plot saved to: {out}")
    return True


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.verbose)
    log = logging.getLogger("mavpose.cli")

    if args.spec:
        sys.exit(0 if redraw_spec(args.log_file, args.spec) else 1)

    from mavpose.chat._deps import ChatDependenciesMissing
    try:
        from mavpose.chat.plot_creator import PlotCreator
    except ChatDependenciesMissing as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    from mavpose.chat.llm import ModelConfigError
    try:
        creator = PlotCreator(max_retries=args.retries, model=args.model,
                              mode="code" if args.code else "spec")
    except ModelConfigError as exc:   # includes MissingAPIKeyError
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"🤖 Model: {creator.model}")

    try:
        creator.set_logfile_name(args.log_file)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    # Phase 1a: schema scan (fast, no DataFrame allocation)
    print(f"\n📂 Parsing log schema: {args.log_file}")
    try:
        creator.parse_mavlink_log()
    except Exception as exc:
        log.error("Failed to parse log file: %s", exc)
        sys.exit(1)
    print(f"✅ Found {len(creator.message_types)} message types.\n")

    if args.prompt:
        try:
            ok = run_once(creator, args.prompt)
        except Exception as exc:
            log.error("Plot generation failed: %s", exc)
            sys.exit(1)
        if not ok:
            sys.exit(1)
    else:
        interactive_mode(creator)


if __name__ == "__main__":
    main()
