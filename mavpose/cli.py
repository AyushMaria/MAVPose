#!/usr/bin/env python3
"""
cli.py

Headless CLI entry point for MAVPose.

Two-phase pipeline
------------------
  Phase 1 — Extraction
    parse_mavlink_log()   : schema scan + ChromaDB embeddings
    find_relevant_data_types() : semantic search → message type names
    extract_dataframes()  : full extraction → clean telemetry.parquet

  Phase 2 — LLM plot generation
    create_plot()         : LLM writes pandas + matplotlib script
    run_script()          : executes script; self-heals on failure

Usage:
    python cli.py <log_file> [options]

Examples:
    python cli.py flight.tlog --prompt "Plot altitude over time"
    python cli.py flight.tlog --prompt "Show battery voltage" --retries 5
    python cli.py flight.tlog           # interactive prompt mode
"""

import argparse
#import json
import logging
import sys

from mavpose.PlotCreator import PlotCreator


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
        help="Path to a MAVLink log file (.tlog, .bin, .log)",
    )
    p.add_argument(
        "--prompt", "-p",
        default=None,
        help="Plot request (e.g. 'Plot altitude over time'). "
             "If omitted, enters interactive prompt mode.",
    )
    p.add_argument(
        "--retries", "-r",
        type=int,
        default=3,
        help="Max self-healing retries on script failure (default: 3)",
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


def run_once(creator: PlotCreator, prompt: str) -> None:
    """Execute one plot request end-to-end."""

    # --- Phase 1: semantic search ---
    print(f"\n🔍 Finding relevant message types for: '{prompt}'")
    msg_types = creator.find_relevant_data_types(prompt)
    print(f"   → {msg_types}")

    # --- Phase 1b: headless extraction → Parquet ---
    print("\n🗃️  Extracting telemetry to Parquet...")
    schema_summary = creator.extract_dataframes(msg_types)
    _print_parquet_summary(schema_summary)
    print(f"   Saved → {creator.parquet_path}")

    # --- Phase 2: LLM writes the script ---
    print(f"\n✍️  Generating plot script with {creator.model}...")
    creator.create_plot(prompt, schema_summary)

    # --- Execute ---
    print("⚙️  Running script...")
    plot_result, code = creator.run_script()

    print("\n" + "─" * 60)
    print("Generated code:")
    print("─" * 60)
    print(code)
    print("─" * 60)

    if creator.plot_path:
        print(f"\n✅ Plot saved to: {creator.plot_path}")
    else:
        print("\n⚠️  Plot file not found — check the generated code above.")


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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.verbose)
    log = logging.getLogger("mavpose.cli")

    creator = PlotCreator(max_retries=args.retries)

    try:
        creator.set_logfile_name(args.log_file)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    # Phase 1a: schema scan + embeddings (fast, no DataFrame allocation)
    print(f"\n📂 Parsing log schema: {args.log_file}")
    try:
        creator.parse_mavlink_log()
    except Exception as exc:
        log.error("Failed to parse log file: %s", exc)
        sys.exit(1)
    print("✅ Schema indexed.\n")

    if args.prompt:
        try:
            run_once(creator, args.prompt)
        except Exception as exc:
            log.error("Plot generation failed: %s", exc)
            sys.exit(1)
    else:
        interactive_mode(creator)


if __name__ == "__main__":
    main()
