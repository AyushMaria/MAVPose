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
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

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
from mavpose.chat.plotspec import PlotSpecError, parse_spec, render_spec  # noqa: E402
from mavpose.chat.safe_executor import execute_script  # noqa: E402
from mavpose.file_validator import VALID_EXTENSIONS, validate_mavlink_file  # noqa: E402
from mavpose.log_extractor import LogExtractor  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = ["PlotCreator", "RunResult", "MissingAPIKeyError", "ModelConfigError", "VALID_EXTENSIONS"]

MODES = ("spec", "code")
FORMATS = ("png", "svg", "pdf")
DEFAULT_DPI = {"spec": 150, "code": 400}
MAX_HISTORY = 3   # earlier requests shown to the model, so follow-ups work

_PYTHON_TAGS = {"python", "py", "python3"}


@dataclass
class RunResult:
    """
    Outcome of :meth:`PlotCreator.run_script`.

    Attributes
    ----------
    success:
        True if the script exited cleanly and saved a new plot.
    plot_path:
        The saved plot, or ``None`` if no plot was created.
    code:
        The last script that ran.
    error:
        The last error, or ``None`` on success.
    attempts:
        How many times the script ran.
    """

    success: bool
    plot_path: Optional[str]
    code: str
    error: Optional[str]
    attempts: int

    def __iter__(self):
        # MAVPose <= 0.3 returned ``([[(None, (plot_path,))]], code)``;
        # keep ``result, code = creator.run_script()`` working for now.
        warnings.warn(
            "Unpacking run_script() is deprecated; use its RunResult attributes "
            "(success, plot_path, code, error) instead.",
            DeprecationWarning, stacklevel=2,
        )
        return iter(([[(None, (self.plot_path,))]], self.code))

SPEC_FORMAT = """{
  "title": "optional figure title",
  "start_s": null, "end_s": null,
  "panels": [
    {"title": "optional", "kind": "line | scatter | step", "y_label": "optional",
     "series": [
       {"msg_type": "EXACT_NAME", "field": "exact_field", "label": "optional",
        "axis": "left | right", "color": "optional matplotlib colour",
        "x_field": "optional field of the same msg_type to use instead of time"}
     ]}
  ]
}"""

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
    Turn a plain-English request into a plot of a flight log.

    In the default ``"spec"`` mode the model describes the plot as JSON (see
    :mod:`mavpose.chat.plotspec`) and MAVPose draws it: no generated code
    runs.  In ``"code"`` mode the model writes a pandas + matplotlib script
    that runs in a restricted subprocess, for plots a spec can't describe.

    Parameters
    ----------
    max_retries:
        Number of self-healing retries when the generated script fails.
    script_timeout:
        Seconds a generated script may run.
    mode:
        ``"spec"`` (default) or ``"code"``.
    dpi:
        Resolution of the saved plot.  Default: 150 in spec mode, 400 in code
        mode.  Ignored for SVG.
    fmt:
        ``"png"`` (default), ``"svg"`` or ``"pdf"``: the plot's file format.
    output_dir:
        Folder for ``plot.*``, ``plot.json``/``plot.py`` and
        ``telemetry.parquet``; created if needed.  Default: the log's folder.
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
        mode: str = "spec",
        dpi: Optional[int] = None,
        fmt: str = "png",
        output_dir: Optional[str] = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        fmt = fmt.lower().lstrip(".")
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}, not {fmt!r}")
        if dpi is not None and (isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0):
            raise ValueError(f"dpi must be a positive whole number, not {dpi!r}")
        self.model_config: ModelConfig = _load_model_config(model)
        self.mode: str = mode
        self.dpi: int = dpi if dpi is not None else DEFAULT_DPI[mode]
        self.fmt: str = fmt
        self.output_dir: Optional[str] = output_dir

        self.logfile_name: str = ""
        self.script_path: str = ""
        self.plot_path: str = ""
        self.parquet_path: str = ""
        self.spec_path: str = ""
        self.last_code: str = ""
        self.last_spec: Optional[dict] = None   # last spec drawn (spec mode)
        # Earlier successful requests in this session, oldest first:
        # [{"request": str, "spec": dict} or {"request": str, "code": str}]
        self.history: List[dict] = []
        self._pending_request: Optional[str] = None   # code mode, until it runs
        self.last_success: bool = False  # did the last request produce a plot?
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
                "parquet_file", "output_file", "dpi",
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
                "- Save the figure to '{output_file}' at dpi={dpi}. Do NOT call plt.show().\n"
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
        # Spec mode: the model describes the plot as JSON; MAVPose draws it.
        # ------------------------------------------------------------------
        spec_prompt = PromptTemplate(
            input_variables=["schema", "history", "human_input", "format", "feedback"],
            template=(
                "You design plots of drone flight-log telemetry. You don't write code: "
                "you describe the plot as JSON and MAVPose draws it.\n\n"
                "Available data (msg_type -> columns, each with dtype, unit and range):\n"
                "{schema}\n\n"
                "Spec format:\n{format}\n\n"
                "Rules:\n"
                "- Use msg_type and field names exactly as listed above.\n"
                "- Values are already in the listed units; axis labels get units automatically.\n"
                "- Series are plotted against time_s (seconds from log start) unless x_field "
                "is set, e.g. x_field \"lon\" with field \"lat\" for a position track.\n"
                "- Put quantities with different units in separate panels, or one on axis "
                "\"right\".\n"
                "- Use start_s / end_s only if the request asks for a time window.\n"
                "- At most 6 panels and 8 series per panel.\n"
                "- Reply with only the JSON object.\n"
                "{history}{feedback}\n"
                "Request: {human_input}"
            ),
        )
        self._spec_chain = spec_prompt | build_chat_model(
            self.model_config, max_tokens=1500
        ) | StrOutputParser()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_code_snippets(text: str) -> list:
        """
        Extract the code blocks worth running from a model reply, best first.

        Blocks fenced as ```` ```python ```` (or ``py``) come first, longest
        first; then unlabelled blocks, longest first.  Blocks labelled with
        another language (e.g. a ```` ```bash ```` install line) are never
        returned.  With no usable fenced block, the whole text is returned.
        """
        blocks = re.findall(r"```([^\n`]*)\n(.*?)```", text, re.DOTALL)
        tagged = [code for tag, code in blocks if tag.strip().lower() in _PYTHON_TAGS]
        untagged = [code for tag, code in blocks if not tag.strip()]
        # sorted() is stable: equally long blocks keep the reply's order
        ranked = (sorted(tagged, key=len, reverse=True)
                  + sorted(untagged, key=len, reverse=True))
        return ranked if ranked else [text]

    @staticmethod
    def write_plot_script(filename: str, text: str) -> None:
        """Write *text* to *filename*."""
        with open(filename, "w") as fh:
            fh.write(text)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_logfile_name(self, filename: str, output_dir: Optional[str] = None) -> None:
        """
        Validate *filename* and register it as the active log file.

        Outputs go to *output_dir*, else the ``output_dir`` given to the
        constructor, else the log's folder.  The folder is created if needed.

        Raises
        ------
        FileNotFoundError
            The file does not exist.
        FileValidationError
            (a ValueError) bad extension, symlink, empty, or too large.
        """
        validate_mavlink_file(filename)
        base_dir = os.path.abspath(
            output_dir or self.output_dir or os.path.dirname(os.path.abspath(filename))
        )
        os.makedirs(base_dir, exist_ok=True)
        if filename != self.logfile_name:   # a different log: earlier plots don't apply
            self.history = []
            self.last_spec = None
            self.last_code = ""
        self.logfile_name = filename
        self.script_path = os.path.join(base_dir, "plot.py")
        self.plot_path = os.path.join(base_dir, f"plot.{self.fmt}")
        self.parquet_path = os.path.join(base_dir, "telemetry.parquet")
        self.spec_path = os.path.join(base_dir, "plot.json")
        self._extractor = LogExtractor(filename)

    # ------------------------------------------------------------------
    # Phase 1a — schema scan
    # ------------------------------------------------------------------

    def parse_mavlink_log(self) -> str:
        """
        Read the log once and collect its message types, fields and row
        counts.  :meth:`find_relevant_data_types` shows these to the model.

        The parsed data is kept, so later requests on the same log don't
        read the file again.

        Returns
        -------
        JSON string of ``{msg_type: {"count", "fields": {name: dtype}, "units"}}``.
        """
        if not self.logfile_name:
            raise RuntimeError("No log file set. Call set_logfile_name() first.")

        frames = self._log_frames()
        units = self._extractor.units
        self.message_types = {
            mt: {
                "count": len(df),
                "fields": {c: str(df[c].dtype) for c in df.columns
                           if c not in ("time_s", "msg_type")},
                "units": {f: u for f, u in units.get(mt, {}).items() if f != "time_s"},
            }
            for mt, df in frames.items()
        }
        return json.dumps(self.message_types, indent=4)

    def _log_frames(self) -> Dict[str, pd.DataFrame]:
        """The log's frames, parsing the file only the first time."""
        if not self._extractor:
            raise RuntimeError("Call set_logfile_name() first.")
        if not self._extractor.frames:
            logger.info("Reading %s", self.logfile_name)
            self._extractor.extract_all()
        return self._extractor.frames

    # ------------------------------------------------------------------
    # Phase 1b — full extraction → Parquet
    # ------------------------------------------------------------------

    def extract_dataframes(
        self,
        msg_types: List[str],
    ) -> Dict[str, dict]:
        """
        Export the given message types to a clean, time-aligned Parquet file.
        The log is parsed on first use and reused for later requests.

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
        self._log_frames()   # parsed once per log, then reused
        logger.info("Exporting message types: %s", msg_types)
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
    # Phase 2 (spec mode) — the model describes the plot, MAVPose draws it
    # ------------------------------------------------------------------

    def _remember(self, entry: dict) -> None:
        self.history = (self.history + [entry])[-MAX_HISTORY:]

    def _history_text(self) -> str:
        """Earlier requests for the prompt: all their specs, but only the latest script."""
        if not self.history:
            return ""
        lines = ["\nEarlier requests in this session (oldest first); the new request "
                 "may refer to them:"]
        for i, entry in enumerate(self.history, 1):
            lines.append(f"{i}. Request: {entry['request']}")
            if "spec" in entry:
                lines.append(f"   Spec: {json.dumps(entry['spec'])}")
            elif i == len(self.history):
                lines.append(f"   Script:\n{entry['code']}")
        return "\n".join(lines) + "\n"

    def plot_from_spec(self, human_input: str, schema_summary: Dict[str, dict]) -> bool:
        """
        Ask the model for a plot spec, check it, and draw it to :attr:`plot_path`.

        If the spec is invalid (e.g. a misspelt field), the error goes back to
        the model, up to ``max_retries`` attempts in all.  The spec that was
        drawn is saved to :attr:`spec_path` and kept in :attr:`last_spec`.
        The last few requests and their specs (:attr:`history`) are shown to
        the model, so follow-ups like "make it red" or "and add GPS altitude"
        work.  Loading a different log clears the history.

        Returns
        -------
        True if a plot was saved (also stored in :attr:`last_success`).
        """
        if not self._extractor or not self._extractor.frames:
            raise RuntimeError("Call extract_dataframes() first.")
        self.last_success = False
        if os.path.lexists(self.plot_path):
            os.remove(self.plot_path)

        history = self._history_text()
        feedback = ""
        reply = ""
        for attempt in range(1, self.max_retries + 1):
            reply = self._spec_chain.invoke({
                "schema": json.dumps(schema_summary, indent=2),
                "history": history,
                "human_input": human_input,
                "format": SPEC_FORMAT,
                "feedback": feedback,
            })
            logger.debug("Spec reply (attempt %d):\n%s", attempt, reply)
            self.last_code = reply.strip()
            try:
                spec = render_spec(parse_spec(reply), self._extractor.frames,
                                   self._extractor.units, self.plot_path, dpi=self.dpi)
            except PlotSpecError as exc:
                logger.warning("Spec attempt %d/%d rejected: %s", attempt, self.max_retries, exc)
                feedback = (f"\nYour previous reply was:\n{reply}\n"
                            f"It could not be drawn: {exc}\nReply with a corrected spec.\n")
                continue
            with open(self.spec_path, "w", encoding="utf-8") as fh:
                json.dump(spec, fh, indent=2)
            self.last_spec = spec
            self.last_code = json.dumps(spec, indent=2)
            self._remember({"request": human_input, "spec": spec})
            self.last_success = True
            return True
        return False

    # ------------------------------------------------------------------
    # Phase 2 (code mode) — LLM plot generation
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
        history = self._history_text()
        schema_text = json.dumps(schema_summary, indent=2)
        response = self._plot_chain.invoke({
            "schema": schema_text,
            "history": history,
            "parquet_file": self.parquet_path,
            "human_input": human_input,
            "output_file": self.plot_path,
            "dpi": self.dpi,
        })
        logger.debug("LLM plot response:\n%s", response)
        code = self.extract_code_snippets(response)
        self.write_plot_script(self.script_path, code[0])
        self.last_code = code[0]
        self._pending_request = human_input
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

    def run_script(self) -> RunResult:
        """
        Execute the generated plot script in the restricted executor.

        The script may only write inside the plot's output directory, runs
        without access to API keys, and is killed after
        ``self.script_timeout`` seconds.  Self-heals up to
        ``self.max_retries`` times on failure.

        A run only counts as successful if the script exits cleanly **and**
        writes a new plot file: any ``plot.png`` left from an earlier run is
        deleted first.  The outcome is also stored in :attr:`last_success`.

        Returns
        -------
        RunResult
        """
        output_dir = os.path.dirname(os.path.abspath(self.plot_path))
        self.last_success = False
        last_error: Optional[str] = None
        attempts = 0
        for attempt in range(1, self.max_retries + 1):
            attempts = attempt
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
                    f"{self.plot_path!r}. It must call plt.savefig({self.plot_path!r}, dpi={self.dpi})."
                    + (f"\n\nScript output:\n{output}" if output.strip() else "")
                )
            if success:
                logger.info("Script succeeded on attempt %d.", attempt)
                self.last_success = True
                self.last_code = code
                last_error = None
                if self._pending_request is not None:
                    self._remember({"request": self._pending_request, "code": code})
                    self._pending_request = None
                break
            last_error = output

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

        return RunResult(
            success=self.last_success,
            plot_path=self.plot_path if self.last_success else None,
            code=self.last_code,
            error=last_error,
            attempts=attempts,
        )
