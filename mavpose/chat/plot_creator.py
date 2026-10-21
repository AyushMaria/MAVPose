"""
mavpose/chat/plot_creator.py

Core MAVPose logic — two-phase pipeline:

  Phase 1 — Extraction (headless DB query)
  -----------------------------------------
  parse_mavlink_log()    : schema-only scan: message types, fields, row counts
  extract_dataframes()   : full extraction of relevant message types
                           → exports a clean Parquet file

  Phase 2 — LLM plot generation
  -----------------------------------------
  find_relevant_data_types() : the model picks the message types it needs
  create_plot()              : LLM writes pandas+matplotlib script
                               against the Parquet file
  run_script()               : executes script; self-heals on failure

The model works with any provider (see mavpose.chat.llm) and never touches
raw binary data.  It receives:
  - The log's list of message types and field names
  - The path of a clean .parquet file
  - Exact column names, dtypes, and value ranges per message type
  - The output path for the .png
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional

from mavpose.chat._deps import require_chat_deps

require_chat_deps()  # friendly error if the [chat] extra is missing

from dotenv import find_dotenv, load_dotenv  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402

from mavpose.chat.llm import (  # noqa: E402
    MissingAPIKeyError,
    ModelConfig,
    ModelConfigError,
    build_chat_model,
    resolve_model,
)
from mavpose.chat.safe_executor import execute_script  # noqa: E402
from mavpose.file_validator import VALID_EXTENSIONS, validate_mavlink_file  # noqa: E402
from mavpose.log_extractor import LogExtractor  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = ["PlotCreator", "MissingAPIKeyError", "ModelConfigError", "VALID_EXTENSIONS"]

# Above this size the message-type catalogue lists names only, so it still
# fits small (e.g. local) models' context windows.
MAX_CATALOG_CHARS = 24_000


def _load_model_config(model: Optional[str]) -> ModelConfig:
    """Resolve the model from *model*, the environment and a .env file."""
    # usecwd=True: search from where the user runs the command.  Without it,
    # python-dotenv searches upward from this installed module's folder and
    # never finds the user's .env.
    load_dotenv(find_dotenv(usecwd=True))
    return resolve_model(model)


def _catalog(message_types: Dict[str, dict]) -> str:
    """The log's message types (and their fields, if that fits) for the model."""
    full = "\n".join(
        f"{mt} ({info.get('count', 0)} rows): {', '.join(info.get('fields', {}))}"
        for mt, info in sorted(message_types.items())
    )
    if len(full) <= MAX_CATALOG_CHARS:
        return full
    return "\n".join(
        f"{mt} ({info.get('count', 0)} rows)" for mt, info in sorted(message_types.items())
    )


def _parse_type_list(response: str, known: List[str]) -> List[str]:
    """Pull the message-type names out of the model's reply, keeping only real ones."""
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    names: List[str] = []
    if match:
        try:
            parsed = json.loads(match.group(0))
            names = [str(n) for n in parsed if isinstance(n, (str, int))]
        except json.JSONDecodeError:
            names = []
    if not names:  # fall back to any known name mentioned in the text
        names = [mt for mt in known if re.search(rf"\b{re.escape(mt)}\b", response)]
    by_lower = {mt.lower(): mt for mt in known}
    chosen = [by_lower[n.strip().lower()] for n in names if n.strip().lower() in by_lower]
    return list(dict.fromkeys(chosen))


class PlotCreator:
    """
    Two-phase pipeline: extract clean Parquet → LLM writes plotting script.

    Parameters
    ----------
    max_retries:
        Number of self-healing retries when the generated script fails.
    script_timeout:
        Seconds a generated script may run.
    model:
        ``provider:model``, e.g. ``"openai:<model>"`` or ``"ollama:llama3.1:8b"``.
        ``None`` uses ``MAVPOSE_MODEL`` or the provider whose API key is set
        (see :mod:`mavpose.chat.llm`).  Settings are also read from a ``.env``
        file in the current folder or a parent folder.

    Raises
    ------
    MissingAPIKeyError
        No API key for the chosen provider, or no provider configured.
    ModelConfigError
        The model setting is incomplete or invalid (``MissingAPIKeyError`` is
        a subclass).
    """

    def __init__(
        self,
        max_retries: int = 3,
        script_timeout: int = 60,
        model: Optional[str] = None,
    ) -> None:
        self.model_config: ModelConfig = _load_model_config(model)

        self.logfile_name: str = ""
        self.script_path: str = ""
        self.plot_path: str = ""
        self.parquet_path: str = ""
        self.last_code: str = ""
        self.last_success: bool = False  # did the last run_script() produce a plot?
        self.message_types: dict = {}   # schema metadata from schema_only()
        self.max_retries: int = max_retries
        self.script_timeout: int = script_timeout
        self._extractor: Optional[LogExtractor] = None

        self.model: str = str(self.model_config)   # "provider:model", for display

        llm = build_chat_model(self.model_config, max_tokens=2000)

        # ------------------------------------------------------------------
        # Plot-generation prompt
        # The LLM receives a Parquet path and exact schema — no binary data.
        # ------------------------------------------------------------------
        plot_prompt = PromptTemplate(
            input_variables=[
                "schema", "history", "human_input",
                "parquet_file", "output_file",
            ],
            template=(
                "You are an expert data-visualisation engineer.\n"
                "You will be given a Parquet file that contains pre-extracted,"
                " time-aligned drone flight-log telemetry. Your task is to write a\n"
                "Python script that reads this file with pandas and plots the"
                " requested data with matplotlib.\n\n"
                "Rules:\n"
                "- Read the data with: df = pd.read_parquet('{parquet_file}')\n"
                "- Filter rows by msg_type when needed, e.g.:\n"
                "    df_gps = df[df['msg_type'] == 'GLOBAL_POSITION_INT']\n"
                "- The 'time_s' column is always float64 seconds from log start,\n"
                "  on one clock shared by every msg_type.\n"
                "- Each column's unit is given in the schema. Values are already\n"
                "  in that unit: do not rescale them. Put units in axis labels.\n"
                "- Plot each independent variable over time_s on its own axis or subplot.\n"
                "- Save the figure to '{output_file}' at dpi=400. Do NOT call plt.show().\n"
                "- Return ONLY the script inside a markdown ```python``` block.\n"
                "- Do NOT import pymavlink, subprocess, os, sys, or socket.\n\n"
                "Available data schema (msg_type → columns with dtype and range):\n"
                "{schema}\n\n"
                "Chat history:\n{history}\n\n"
                "HUMAN: {human_input}"
            ),
        )
        self._plot_chain = plot_prompt | llm | StrOutputParser()

        # ------------------------------------------------------------------
        # Fix prompt — also operates on the Parquet / pandas context
        # ------------------------------------------------------------------
        fix_prompt = PromptTemplate(
            input_variables=["schema", "error", "script", "parquet_file"],
            template=(
                "You are debugging a pandas + matplotlib script that reads"
                " flight-log telemetry from a Parquet file.\n\n"
                "Parquet file: {parquet_file}\n"
                "Schema:\n{schema}\n\n"
                "Failing script:\n{script}\n\n"
                "Error:\n{error}\n\n"
                "Fix the script. Return ONLY the corrected script in a"
                " markdown ```python``` block."
            ),
        )
        fix_llm = build_chat_model(self.model_config, max_tokens=8000)
        self._fix_chain = fix_prompt | fix_llm | StrOutputParser()

        # ------------------------------------------------------------------
        # Message-type selection: the model reads the log's catalogue of
        # message types and picks the ones the request needs.
        # ------------------------------------------------------------------
        select_prompt = PromptTemplate(
            input_variables=["catalog", "human_input"],
            template=(
                "A drone flight log contains these message types "
                "(name, row count, and fields):\n\n{catalog}\n\n"
                "Which message types are needed to answer this request?\n"
                "Request: {human_input}\n\n"
                "Choose every type needed (for example, both GPS and barometer types "
                "for a GPS-vs-baro comparison), but no unrelated ones. Reply with only "
                "a JSON array of exact names from the list, e.g. [\"ATTITUDE\", \"VFR_HUD\"]."
            ),
        )
        self._select_chain = select_prompt | build_chat_model(
            self.model_config, max_tokens=500
        ) | StrOutputParser()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_code_snippets(text: str) -> list:
        """
        Extract fenced ```python``` blocks from *text*.
        Falls back to returning [text] when no fences are found.
        """
        snippets = re.findall(r"```.*?\n(.*?)```", text, re.DOTALL | re.MULTILINE)
        return snippets if snippets else [text]

    @staticmethod
    def write_plot_script(filename: str, text: str) -> None:
        """Write *text* to *filename*."""
        with open(filename, "w") as fh:
            fh.write(text)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_logfile_name(self, filename: str) -> None:
        """
        Validate *filename* and register it as the active log file.

        Raises
        ------
        FileNotFoundError
            The file does not exist.
        FileValidationError
            (a ValueError) bad extension, symlink, empty, or too large.
        """
        validate_mavlink_file(filename)
        base_dir = os.path.dirname(os.path.abspath(filename))
        self.logfile_name = filename
        self.script_path = os.path.join(base_dir, "plot.py")
        self.plot_path = os.path.join(base_dir, "plot.png")
        self.parquet_path = os.path.join(base_dir, "telemetry.parquet")
        self._extractor = LogExtractor(filename)

    # ------------------------------------------------------------------
    # Phase 1a — schema scan
    # ------------------------------------------------------------------

    def parse_mavlink_log(self) -> str:
        """
        Schema-only scan: collect the message types, their fields and row
        counts.  :meth:`find_relevant_data_types` shows this to the model.

        Returns
        -------
        JSON string of all discovered message types and their fields.
        """
        if not self.logfile_name:
            raise RuntimeError("No log file set. Call set_logfile_name() first.")

        self.message_types = self._extractor.schema_only()
        return json.dumps(self.message_types, indent=4)

    # ------------------------------------------------------------------
    # Phase 1b — full extraction → Parquet
    # ------------------------------------------------------------------

    def extract_dataframes(
        self,
        msg_types: List[str],
    ) -> Dict[str, dict]:
        """
        Run the full extraction for the given message types and export a
        clean, time-aligned Parquet file.

        This is the "headless DB query" step.  The parent process
        (cli.py) calls this after find_relevant_data_types() resolves
        which message types are needed, and before invoking the LLM.

        Parameters
        ----------
        msg_types:
            List of MAVLink message type names identified as relevant.

        Returns
        -------
        Schema summary dict (msg_type → rows/columns/dtype/min/max)
        suitable for direct injection into the LLM prompt.
        """
        if not self._extractor:
            raise RuntimeError("Call set_logfile_name() first.")

        logger.info("Running full extraction for types: %s", msg_types)
        self._extractor.extract_all()
        schema_summary = self._extractor.export_parquet(msg_types, self.parquet_path)
        return schema_summary

    # ------------------------------------------------------------------
    # Choosing message types
    # ------------------------------------------------------------------

    def find_relevant_data_types(self, human_input: str) -> List[str]:
        """
        Ask the model which of the log's message types the request needs.

        Returns
        -------
        Message type names, all present in the log.

        Raises
        ------
        RuntimeError
            :meth:`parse_mavlink_log` has not been called.
        ValueError
            The model did not name any message type in the log.
        """
        if not self.message_types:
            raise RuntimeError("No log schema. Call parse_mavlink_log() first.")
        known = list(self.message_types)
        response = self._select_chain.invoke({
            "catalog": _catalog(self.message_types),
            "human_input": human_input,
        })
        logger.debug("Message-type selection response:\n%s", response)
        chosen = _parse_type_list(response, known)
        if not chosen:
            raise ValueError(
                "Couldn't tell which message types this request needs. Try naming "
                "the data more specifically (e.g. 'battery voltage', 'GPS altitude'), "
                "or run with --verbose to see the model's answer."
            )
        return chosen

    # ------------------------------------------------------------------
    # Phase 2 — LLM plot generation
    # ------------------------------------------------------------------

    def create_plot(
        self,
        human_input: str,
        schema_summary: Dict[str, dict],
    ) -> str:
        """
        Ask the LLM to write a pandas + matplotlib plotting script
        against the exported Parquet file.

        Parameters
        ----------
        human_input:
            The user's natural-language plot request.
        schema_summary:
            Dict returned by extract_dataframes() — msg_type →
            {rows, columns: {col: {dtype, min, max}}}.

        Returns
        -------
        The generated script as a string (also written to disk).
        """
        history = (
            f"\n\nLast script generated:\n\n{self.last_code}" if self.last_code else ""
        )
        schema_text = json.dumps(schema_summary, indent=2)
        response = self._plot_chain.invoke({
            "schema": schema_text,
            "history": history,
            "parquet_file": self.parquet_path,
            "human_input": human_input,
            "output_file": self.plot_path,
        })
        logger.debug("LLM plot response:\n%s", response)
        code = self.extract_code_snippets(response)
        self.write_plot_script(self.script_path, code[0])
        self.last_code = code[0]
        return code[0]

    # ------------------------------------------------------------------
    # Script execution with self-healing
    # ------------------------------------------------------------------

    def attempt_to_fix_script(self, error_message: str) -> str:
        """
        Feed the error back to the LLM and rewrite the failing script.

        Parameters
        ----------
        error_message:
            The stderr / exception text from the failed run.

        Returns
        -------
        The fixed (or best-effort) script as a string.
        """
        with open(self.script_path, "r") as fh:
            script = fh.read()

        # Schema summary (with units) from the cached extraction
        schema_text = "{}"
        if self._extractor and self._extractor.frames:
            try:
                schema_text = json.dumps(self._extractor.summary(), indent=2)
            except Exception as exc:
                logger.warning("Could not rebuild schema for fix prompt: %s", exc)

        try:
            response = self._fix_chain.invoke({
                "schema": schema_text,
                "parquet_file": self.parquet_path,
                "error": error_message,
                "script": script,
            })
        except Exception as exc:
            logger.error("LLM fix request failed: %s", exc)
            return script

        logger.debug("LLM fix response:\n%s", response)
        fixed = self.extract_code_snippets(response)[0]
        self.write_plot_script(self.script_path, fixed)
        return fixed

    def run_script(self) -> tuple:
        """
        Execute the generated plot script in the restricted executor.

        The script may only write inside the plot's output directory, runs
        without access to API keys, and is killed after
        ``self.script_timeout`` seconds.  Self-heals up to
        ``self.max_retries`` times on failure.

        A run only counts as successful if the script exits cleanly **and**
        writes a new plot file: any ``plot.png`` left from an earlier run is
        deleted first.  The outcome is stored in :attr:`last_success`.

        Returns
        -------
        ([(None, (plot_path,))], last_code)
        """
        output_dir = os.path.dirname(os.path.abspath(self.plot_path))
        self.last_success = False
        for attempt in range(1, self.max_retries + 1):
            with open(self.script_path, "r", encoding="utf-8") as fh:
                code = fh.read()

            # A plot from a previous run must not be mistaken for this one.
            if os.path.lexists(self.plot_path):
                os.remove(self.plot_path)

            success, output = execute_script(
                code,
                timeout_seconds=self.script_timeout,
                allowed_write_dirs=[output_dir],
            )
            if success and not os.path.isfile(self.plot_path):
                success = False
                output = (
                    f"The script ran without errors but did not save the figure to "
                    f"{self.plot_path!r}. It must call plt.savefig({self.plot_path!r}, dpi=400)."
                    + (f"\n\nScript output:\n{output}" if output.strip() else "")
                )
            if success:
                logger.info("Script succeeded on attempt %d.", attempt)
                self.last_success = True
                break

            logger.warning(
                "Script attempt %d/%d failed:\n%s",
                attempt, self.max_retries, output,
            )
            if attempt < self.max_retries:
                self.last_code = self.attempt_to_fix_script(output)
            else:
                logger.error("All %d attempts exhausted.", self.max_retries)
                self.last_code = (
                    f"# All {self.max_retries} fix attempts failed.\n"
                    f"# Last error:\n# {output}\n\n"
                    + self.last_code
                )

        return [[(None, (self.plot_path,))]], self.last_code
