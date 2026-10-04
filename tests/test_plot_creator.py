"""
tests/test_plot_creator.py

Tests for PlotCreator's file validation and sandboxed script execution.
The LLM chains are never invoked; ChatOpenAI is constructed but makes no
network calls until used.
"""

import os
import sys
from unittest.mock import patch

import pytest

from mavpose.file_validator import FileValidationError

pytest.importorskip("langchain_openai", reason="needs the [chat] extra")

from mavpose.chat.plot_creator import PlotCreator  # noqa: E402


@pytest.fixture
def creator(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-placeholder")
    return PlotCreator(max_retries=2, script_timeout=30)


@pytest.fixture
def log_file(tmp_path):
    f = tmp_path / "flight.tlog"
    f.write_bytes(b"\xfe\x09\x00\x01\x01\x00")
    return f


class TestSetLogfileName:

    def test_accepts_valid_file(self, creator, log_file):
        creator.set_logfile_name(str(log_file))
        assert creator.plot_path == str(log_file.parent / "plot.png")

    def test_rejects_missing_file(self, creator, tmp_path):
        with pytest.raises(FileNotFoundError):
            creator.set_logfile_name(str(tmp_path / "nope.tlog"))

    def test_rejects_bad_extension(self, creator, tmp_path):
        f = tmp_path / "flight.txt"
        f.write_text("x")
        with pytest.raises(FileValidationError):
            creator.set_logfile_name(str(f))

    def test_rejects_empty_file(self, creator, tmp_path):
        f = tmp_path / "empty.tlog"
        f.touch()
        with pytest.raises(FileValidationError):
            creator.set_logfile_name(str(f))

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin on Windows")
    def test_rejects_symlink(self, creator, log_file, tmp_path):
        link = tmp_path / "link.tlog"
        os.symlink(log_file, link)
        with pytest.raises(FileValidationError):
            creator.set_logfile_name(str(link))


class TestRunScript:

    def _write(self, creator, code):
        with open(creator.script_path, "w", encoding="utf-8") as fh:
            fh.write(code)
        creator.last_code = code

    def test_successful_script_writes_plot(self, creator, log_file):
        creator.set_logfile_name(str(log_file))
        self._write(creator, f"open({creator.plot_path!r}, 'wb').write(b'png')\n")
        with patch.object(creator, "attempt_to_fix_script") as fix:
            creator.run_script()
        fix.assert_not_called()
        assert os.path.exists(creator.plot_path)

    def test_malicious_script_is_blocked_and_sent_for_fixing(self, creator, log_file):
        creator.set_logfile_name(str(log_file))
        self._write(creator, "import pandas as pd\npd.io.common.os.system('echo ' + 'ESC' + 'APED')\n")
        with patch.object(creator, "attempt_to_fix_script", return_value="pass") as fix:
            creator.run_script()
        fix.assert_called_once()
        error_text = fix.call_args[0][0]
        assert "Blocked by MAVPose sandbox" in error_text
        assert "ESCAPED" not in error_text

    def test_script_cannot_write_outside_output_dir(self, creator, log_file, tmp_path_factory):
        creator.set_logfile_name(str(log_file))
        outside = tmp_path_factory.mktemp("elsewhere") / "pwn.txt"
        self._write(creator, f"open({str(outside)!r}, 'w').write('x')\n")
        with patch.object(creator, "attempt_to_fix_script", return_value="pass"):
            creator.run_script()
        assert not outside.exists()

    def test_gives_up_after_max_retries(self, creator, log_file):
        creator.set_logfile_name(str(log_file))
        self._write(creator, "raise RuntimeError('always fails')\n")
        with patch.object(creator, "attempt_to_fix_script",
                          return_value="raise RuntimeError('always fails')\n") as fix:
            _, code = creator.run_script()
        assert fix.call_count == creator.max_retries - 1
        assert "fix attempts failed" in code


class TestFixPromptSchema:

    def test_fix_prompt_schema_includes_units(self, creator, tmp_path):
        from unittest.mock import MagicMock

        from tests.log_builders import build_tlog

        log = build_tlog(tmp_path / "flight.tlog")
        creator.set_logfile_name(str(log))
        creator.extract_dataframes(["GLOBAL_POSITION_INT"])
        with open(creator.script_path, "w", encoding="utf-8") as fh:
            fh.write("raise ValueError('boom')\n")

        creator._fix_chain = MagicMock()
        creator._fix_chain.invoke.return_value = "```python\npass\n```"
        creator.attempt_to_fix_script("ValueError: boom")

        schema = creator._fix_chain.invoke.call_args[0][0]["schema"]
        assert '"unit": "m"' in schema
        assert '"GLOBAL_POSITION_INT"' in schema


class TestUlogInChat:

    def test_set_logfile_accepts_ulg(self, creator, tmp_path):
        from tests.log_builders import build_ulog

        log = build_ulog(tmp_path / "flight.ulg")
        creator.set_logfile_name(str(log))
        summary = creator.extract_dataframes(["battery_status"])
        assert summary["battery_status"]["columns"]["voltage_v"]["unit"] == "V"
