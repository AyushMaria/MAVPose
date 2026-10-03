"""
mavpose/PlotCreator.py

Core MAVPose logic — two-phase pipeline:

  Phase 1 — Extraction (headless DB query)
  -----------------------------------------
  parse_mavlink_log()    : schema-only scan → builds ChromaDB embeddings
  extract_dataframes()   : full extraction of relevant message types
                           → exports a clean Parquet file

  Phase 2 — LLM plot generation
  -----------------------------------------
  find_relevant_data_types() : semantic search → relevant msg-type schema
  create_plot()              : LLM writes pandas+matplotlib script
                               against the Parquet file
  run_script()               : executes script; self-heals on failure

The LLM never touches raw binary data.  It receives:
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

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from mavpose.file_validator import VALID_EXTENSIONS, validate_mavlink_file
from mavpose.log_extractor import LogExtractor
from mavpose.safe_executor import execute_script

logger = logging.getLogger(__name__)

__all__ = ["PlotCreator", "VALID_EXTENSIONS"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-5.1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"


class PlotCreator:
    """
    Two-phase pipeline: extract clean Parquet → LLM writes plotting script.

    Parameters
    ----------
    max_retries:
        Number of self-healing retries when the generated script fails.
    """

    def __init__(self, max_retries: int = 3, script_timeout: int = 60) -> None:
        load_dotenv()

        self.logfile_name: str = ""
        self.script_path: str = ""
        self.plot_path: str = ""
        self.parquet_path: str = ""
        self.last_code: str = ""
        self.message_types: dict = {}   # schema metadata from schema_only()
        self.db: Optional[Chroma] = None
        self.max_retries: int = max_retries
        self.script_timeout: int = script_timeout
        self._extractor: Optional[LogExtractor] = None

        api_key: str = os.environ["OPENROUTER_API_KEY"]
        self.model: str = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

        llm = ChatOpenAI(
            model_name=self.model,
            max_tokens=2000,
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_BASE_URL,
        )

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
                " time-aligned MAVLink telemetry. Your task is to write a\n"
                "Python script that reads this file with pandas and plots the"
                " requested data with matplotlib.\n\n"
                "Rules:\n"
                "- Read the data with: df = pd.read_parquet('{parquet_file}')\n"
                "- Filter rows by msg_type when needed, e.g.:\n"
                "    df_gps = df[df['msg_type'] == 'GLOBAL_POSITION_INT']\n"
                "- The 'time_s' column is always float64 seconds from log start.\n"
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
                " MAVLink telemetry from a Parquet file.\n\n"
                "Parquet file: {parquet_file}\n"
                "Schema:\n{schema}\n\n"
                "Failing script:\n{script}\n\n"
                "Error:\n{error}\n\n"
                "Fix the script. Return ONLY the corrected script in a"
                " markdown ```python``` block."
            ),
        )
        fix_llm = ChatOpenAI(
            model_name=self.model,
            max_tokens=8000,
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_BASE_URL,
        )
        self._fix_chain = fix_prompt | fix_llm | StrOutputParser()

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
    # Phase 1a — schema scan + embeddings
    # ------------------------------------------------------------------

    def parse_mavlink_log(self) -> str:
        """
        Schema-only scan: collect message-type metadata and build the
        ChromaDB vector index used for semantic field search.

        Returns
        -------
        JSON string of all discovered message types and their fields.
        """
        if not self.logfile_name:
            raise RuntimeError("No log file set. Call set_logfile_name() first.")

        self.message_types = self._extractor.schema_only()
        self._create_embeddings(self.message_types)
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
    # Embeddings
    # ------------------------------------------------------------------

    def _create_embeddings(self, message_types: dict) -> None:
        """
        Embed each message type into a persisted ChromaDB collection
        using OpenRouter's OpenAI-compatible embeddings endpoint.
        """
        texts = [json.dumps({mt: message_types[mt]}) for mt in message_types]
        logger.info("Creating embeddings for %d message types.", len(texts))

        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base=OPENROUTER_BASE_URL,
        )
        persist_dir = os.path.join(
            os.path.dirname(os.path.abspath(self.logfile_name)), "chroma_db"
        )
        self.db = Chroma.from_texts(
            texts,
            embeddings,
            persist_directory=persist_dir,
        )
        logger.info("ChromaDB persisted to %s", persist_dir)

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def find_relevant_data_types(self, human_input: str) -> List[str]:
        """
        Semantic similarity search against the ChromaDB vector store.

        Returns
        -------
        List of MAVLink message type names most relevant to *human_input*.
        """
        if self.db is None:
            raise RuntimeError("Vector store not initialised. Call parse_mavlink_log() first.")
        docs = self.db.similarity_search(human_input)
        # Each document was stored as JSON of a single message type dict;
        # parse back to extract the type name.
        types: List[str] = []
        for doc in docs:
            try:
                obj = json.loads(doc.page_content)
                types.extend(obj.keys())
            except (json.JSONDecodeError, AttributeError):
                pass
        return list(dict.fromkeys(types))  # deduplicate, preserve order

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

        # Rebuild schema summary from cached extractor frames
        schema_text = "{}"
        if self._extractor and self._extractor.frames:
            try:
                summary: Dict[str, dict] = {}
                for mt, df in self._extractor.frames.items():
                    cols = {}
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        if pd.api.types.is_numeric_dtype(df[col]):
                            cols[col] = {
                                "dtype": dtype,
                                "min": round(float(df[col].min()), 6),
                                "max": round(float(df[col].max()), 6),
                            }
                        else:
                            cols[col] = {"dtype": dtype}
                    summary[mt] = {"rows": len(df), "columns": cols}
                schema_text = json.dumps(summary, indent=2)
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

        Returns
        -------
        ([(None, (plot_path,))], last_code)
        """
        output_dir = os.path.dirname(os.path.abspath(self.plot_path))
        for attempt in range(1, self.max_retries + 1):
            with open(self.script_path, "r", encoding="utf-8") as fh:
                code = fh.read()

            success, output = execute_script(
                code,
                timeout_seconds=self.script_timeout,
                allowed_write_dirs=[output_dir],
            )
            if success:
                logger.info("Script succeeded on attempt %d.", attempt)
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
