"""
tests/test_plot_creator.py

Tests for PlotCreator's file validation and sandboxed script execution.
The LLM chains are never invoked; ChatOpenAI is constructed but makes no
network calls until used.
"""

import json
import os
import sys
from unittest.mock import patch

import pytest

from mavpose.file_validator import FileValidationError

pytest.importorskip("langchain_openai", reason="needs the [chat] extra")

from mavpose.chat.plot_creator import MissingAPIKeyError, PlotCreator  # noqa: E402


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
        assert creator.last_success is True

    def test_script_that_saves_nothing_is_a_failure(self, creator, log_file):
        """#9: a clean exit without a plot file must not count as success."""
        creator.set_logfile_name(str(log_file))
        self._write(creator, "x = 1\n")
        with patch.object(creator, "attempt_to_fix_script", return_value="x = 1\n") as fix:
            creator.run_script()
        assert creator.last_success is False
        assert fix.call_count == creator.max_retries - 1
        assert "did not save the figure" in fix.call_args[0][0]

    def test_stale_plot_from_previous_run_is_not_success(self, creator, log_file):
        """#9: an old plot.png must not make a failed run look successful."""
        creator.set_logfile_name(str(log_file))
        with open(creator.plot_path, "wb") as fh:
            fh.write(b"old png")
        self._write(creator, "raise RuntimeError('boom')\n")
        with patch.object(creator, "attempt_to_fix_script",
                          return_value="raise RuntimeError('boom')\n"):
            creator.run_script()
        assert creator.last_success is False
        assert not os.path.exists(creator.plot_path)

    def test_success_after_a_failed_run_resets_flag(self, creator, log_file):
        creator.set_logfile_name(str(log_file))
        self._write(creator, "raise RuntimeError('boom')\n")
        with patch.object(creator, "attempt_to_fix_script",
                          return_value="raise RuntimeError('boom')\n"):
            creator.run_script()
        assert creator.last_success is False
        self._write(creator, f"open({creator.plot_path!r}, 'wb').write(b'png')\n")
        creator.run_script()
        assert creator.last_success is True

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


class TestApiKey:
    """#15: a missing key gives a clear error, and .env is read from the working folder."""

    def test_missing_key_raises_friendly_error(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY is not set"):
            PlotCreator()

    def test_empty_key_counts_as_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "  ")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(MissingAPIKeyError):
            PlotCreator()

    def test_key_is_read_from_dotenv_in_working_folder(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-from-dotenv\n")
        workdir = tmp_path / "logs"
        workdir.mkdir()
        monkeypatch.chdir(workdir)          # .env is found in a parent folder too
        creator = PlotCreator()
        assert creator._api_key == "sk-from-dotenv"

    def test_shell_variable_wins_over_dotenv(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-shell")
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-from-dotenv\n")
        monkeypatch.chdir(tmp_path)
        assert PlotCreator()._api_key == "sk-from-shell"


class TestVectorStoreIsolation:
    """#10: two logs in one folder must not share a vector index."""

    @pytest.fixture
    def fake_embeddings(self, monkeypatch):
        from langchain_core.embeddings import DeterministicFakeEmbedding

        import mavpose.chat.plot_creator as pc
        monkeypatch.setattr(pc, "OpenAIEmbeddings",
                            lambda **kw: DeterministicFakeEmbedding(size=16))

    def test_two_logs_in_one_folder_keep_separate_indexes(
        self, monkeypatch, tmp_path, fake_embeddings
    ):
        from tests.log_builders import build_dataflash_bin, build_tlog

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-placeholder")
        tlog = build_tlog(tmp_path / "a.tlog")
        dfl = build_dataflash_bin(tmp_path / "b.bin")

        first = PlotCreator()
        first.set_logfile_name(str(tlog))
        first.parse_mavlink_log()
        second = PlotCreator()
        second.set_logfile_name(str(dfl))
        second.parse_mavlink_log()

        for creator in (first, second):
            stored = creator.db.get()["documents"]
            names = {name for doc in stored for name in json.loads(doc)}
            assert names == set(creator.message_types)
        assert not set(first.message_types) & set(second.message_types)
        assert not (tmp_path / "chroma_db").exists()

    def test_reparsing_does_not_duplicate_entries(self, monkeypatch, tmp_path, fake_embeddings):
        from tests.log_builders import build_tlog

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-placeholder")
        creator = PlotCreator()
        creator.set_logfile_name(str(build_tlog(tmp_path / "a.tlog")))
        creator.parse_mavlink_log()
        old_name = creator.db._collection.name
        creator.parse_mavlink_log()
        assert len(creator.db.get()["documents"]) == len(creator.message_types)
        # the previous index is dropped, not left in memory
        names = [getattr(c, "name", c) for c in creator.db._client.list_collections()]
        assert old_name not in names
        assert creator.db._collection.name in names


class TestCliOutcome:
    """#9: the CLI reports what actually happened."""

    def _creator(self, success):
        from unittest.mock import MagicMock

        c = MagicMock()
        c.find_relevant_data_types.return_value = ["ATTITUDE"]
        c.extract_dataframes.return_value = {}
        c.run_script.return_value = ([[(None, ("plot.png",))]], "code")
        c.last_success = success
        c.max_retries = 3
        c.plot_path = "/logs/plot.png"
        return c

    def test_failure_is_reported(self, capsys):
        from mavpose.chat.cli import run_once

        assert run_once(self._creator(False), "plot it") is False
        out = capsys.readouterr().out
        assert "No plot was created" in out
        assert "Plot saved" not in out

    def test_success_is_reported(self, capsys):
        from mavpose.chat.cli import run_once

        assert run_once(self._creator(True), "plot it") is True
        assert "Plot saved to: /logs/plot.png" in capsys.readouterr().out

    def test_missing_key_exits_cleanly(self, monkeypatch, tmp_path, capsys):
        from mavpose.chat import cli

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["mavpose", "flight.tlog", "-p", "x"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "OPENROUTER_API_KEY is not set" in err
        assert "Traceback" not in err
