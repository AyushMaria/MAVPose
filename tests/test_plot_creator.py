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


MODEL_VARS = ["MAVPOSE_MODEL", "MAVPOSE_API_KEY", "MAVPOSE_BASE_URL", "OPENROUTER_MODEL",
              "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
              "GOOGLE_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY",
              "TOGETHER_API_KEY"]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No provider settings from the machine running the tests, and no .env."""
    for var in MODEL_VARS:
        monkeypatch.delenv(var, raising=False)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return monkeypatch


@pytest.fixture
def creator(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "sk-test-placeholder")
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
    """A missing key gives a clear error, and .env is read from the working folder."""

    def test_no_provider_raises_friendly_error(self, clean_env):
        with pytest.raises(MissingAPIKeyError, match="No AI provider is configured"):
            PlotCreator()

    def test_empty_key_counts_as_missing(self, clean_env):
        clean_env.setenv("OPENROUTER_API_KEY", "  ")
        with pytest.raises(MissingAPIKeyError):
            PlotCreator()

    def test_key_is_read_from_dotenv_in_working_folder(self, clean_env, tmp_path):
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-from-dotenv\n")
        # clean_env runs us in tmp_path/cwd: .env is found in a parent folder too
        creator = PlotCreator()
        assert creator.model_config.api_key == "sk-from-dotenv"
        assert creator.model == "openrouter:z-ai/glm-5.1"

    def test_shell_variable_wins_over_dotenv(self, clean_env, tmp_path):
        clean_env.setenv("OPENROUTER_API_KEY", "sk-from-shell")
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-from-dotenv\n")
        assert PlotCreator().model_config.api_key == "sk-from-shell"

    def test_model_setting_is_read_from_dotenv(self, clean_env, tmp_path):
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant\nMAVPOSE_MODEL=anthropic:test-model\n")
        creator = PlotCreator()
        assert creator.model == "anthropic:test-model"
        assert type(creator._select_chain.steps[1]).__name__ == "ChatAnthropic"

    def test_model_argument_wins(self, clean_env):
        clean_env.setenv("OPENROUTER_API_KEY", "sk-or")
        clean_env.setenv("MAVPOSE_MODEL", "openrouter:from-env")
        assert PlotCreator(model="ollama:llama3.1:8b").model == "ollama:llama3.1:8b"


class TestMessageTypeSelection:
    """#12: the model picks message types from the log's own list (no vector search)."""

    def _creator_with_log(self, creator, tmp_path):
        from tests.log_builders import build_tlog

        creator.set_logfile_name(str(build_tlog(tmp_path / "a.tlog")))
        creator.parse_mavlink_log()
        return creator

    def _answer(self, creator, text):
        from unittest.mock import MagicMock

        creator._select_chain = MagicMock()
        creator._select_chain.invoke.return_value = text
        return creator._select_chain

    def test_model_sees_every_message_type_and_its_fields(self, creator, tmp_path):
        c = self._creator_with_log(creator, tmp_path)
        chain = self._answer(c, '["GLOBAL_POSITION_INT"]')
        c.find_relevant_data_types("plot altitude")
        sent = chain.invoke.call_args[0][0]
        assert sent["human_input"] == "plot altitude"
        for mt in c.message_types:
            assert mt in sent["catalog"]
        assert "relative_alt" in sent["catalog"]

    def test_several_types_are_kept_in_order(self, creator, tmp_path):
        c = self._creator_with_log(creator, tmp_path)
        self._answer(c, 'Sure:\n```json\n["SYS_STATUS", "GLOBAL_POSITION_INT"]\n```')
        assert c.find_relevant_data_types("battery and altitude") == [
            "SYS_STATUS", "GLOBAL_POSITION_INT"]

    def test_names_not_in_the_log_are_dropped_and_case_is_fixed(self, creator, tmp_path):
        c = self._creator_with_log(creator, tmp_path)
        self._answer(c, '["global_position_int", "MADE_UP", "GLOBAL_POSITION_INT"]')
        assert c.find_relevant_data_types("x") == ["GLOBAL_POSITION_INT"]

    def test_plain_text_answer_still_works(self, creator, tmp_path):
        c = self._creator_with_log(creator, tmp_path)
        self._answer(c, "You need SYS_STATUS for that.")
        assert c.find_relevant_data_types("battery") == ["SYS_STATUS"]

    def test_no_usable_answer_is_a_clear_error(self, creator, tmp_path):
        c = self._creator_with_log(creator, tmp_path)
        self._answer(c, "[]")
        with pytest.raises(ValueError, match="Couldn't tell which message types"):
            c.find_relevant_data_types("the weather")

    def test_needs_a_parsed_log(self, creator):
        with pytest.raises(RuntimeError, match="parse_mavlink_log"):
            creator.find_relevant_data_types("x")

    def test_large_catalog_falls_back_to_names(self, monkeypatch):
        import mavpose.chat.plot_creator as pc

        types = {f"TOPIC_{i}": {"count": 1, "fields": {f"field_{j}": "float" for j in range(50)}}
                 for i in range(100)}
        monkeypatch.setattr(pc, "MAX_CATALOG_CHARS", 5000)
        text = pc._catalog(types)
        assert "TOPIC_99" in text and "field_0" not in text

    def test_no_files_written_for_the_index(self, creator, tmp_path):
        self._creator_with_log(creator, tmp_path)
        assert not (tmp_path / "chroma_db").exists()


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
        c.mode = "code"
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

    def test_missing_key_exits_cleanly(self, clean_env, capsys):
        from mavpose.chat import cli

        clean_env.setattr(sys, "argv", ["mavpose", "flight.tlog", "-p", "x"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "No AI provider is configured" in err
        assert "Traceback" not in err

    def test_model_flag_reaches_the_creator(self, clean_env, capsys, tmp_path):
        from mavpose.chat import cli

        clean_env.setattr(sys, "argv", ["mavpose", str(tmp_path / "nope.tlog"),
                                        "--model", "ollama:llama3.1:8b"])
        with pytest.raises(SystemExit):
            cli.main()      # stops at the missing log, after building the creator
        assert "Model: ollama:llama3.1:8b" in capsys.readouterr().out

    def test_bad_model_setting_exits_cleanly(self, clean_env, capsys):
        from mavpose.chat import cli

        clean_env.setattr(sys, "argv", ["mavpose", "f.tlog", "--model", "anthropc:x"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert "Did you mean 'anthropic'" in capsys.readouterr().err


class TestSpecMode:
    """R7: the model describes the plot as JSON and MAVPose draws it."""

    GOOD = '{"title": "Alt", "panels": [{"series": [{"msg_type": "GLOBAL_POSITION_INT", "field": "relative_alt"}]}]}'
    BAD = '{"panels": [{"series": [{"msg_type": "GLOBAL_POSITION_INT", "field": "altitude"}]}]}'

    def _ready(self, creator, tmp_path, replies):
        from unittest.mock import MagicMock

        from tests.log_builders import build_tlog

        creator.set_logfile_name(str(build_tlog(tmp_path / "f.tlog")))
        creator.parse_mavlink_log()
        schema = creator.extract_dataframes(["GLOBAL_POSITION_INT"])
        creator._spec_chain = MagicMock()
        creator._spec_chain.invoke.side_effect = list(replies)
        return schema

    def test_is_the_default_mode(self, creator):
        assert creator.mode == "spec"

    def test_rejects_unknown_mode(self, clean_env):
        clean_env.setenv("OPENROUTER_API_KEY", "k")
        with pytest.raises(ValueError, match="mode must be"):
            PlotCreator(mode="magic")

    def test_good_spec_is_drawn_and_saved(self, creator, tmp_path):
        schema = self._ready(creator, tmp_path, [f"```json\n{self.GOOD}\n```"])
        assert creator.plot_from_spec("plot altitude", schema) is True
        assert creator.last_success
        assert open(creator.plot_path, "rb").read(4) == b"\x89PNG"
        saved = json.load(open(creator.spec_path))
        assert saved["panels"][0]["series"][0]["field"] == "relative_alt"
        assert creator.last_spec == saved

    def test_no_code_is_written_or_run(self, creator, tmp_path):
        schema = self._ready(creator, tmp_path, [self.GOOD])
        with patch("mavpose.chat.plot_creator.execute_script") as run:
            creator.plot_from_spec("plot altitude", schema)
        run.assert_not_called()
        assert not os.path.exists(creator.script_path)

    def test_invalid_spec_is_sent_back_with_the_error(self, creator, tmp_path):
        schema = self._ready(creator, tmp_path, [self.BAD, self.GOOD])
        assert creator.plot_from_spec("plot altitude", schema) is True
        second = creator._spec_chain.invoke.call_args_list[1][0][0]
        assert "No field 'altitude'" in second["feedback"]
        assert "relative_alt" in second["feedback"]   # the valid fields are listed

    def test_gives_up_after_max_retries_and_leaves_no_plot(self, creator, tmp_path):
        schema = self._ready(creator, tmp_path, [self.BAD] * creator.max_retries)
        with open(creator.plot_path, "wb") as fh:
            fh.write(b"old")
        assert creator.plot_from_spec("plot altitude", schema) is False
        assert creator._spec_chain.invoke.call_count == creator.max_retries
        assert not creator.last_success
        assert not os.path.exists(creator.plot_path)

    def test_follow_up_sees_the_previous_spec(self, creator, tmp_path):
        schema = self._ready(creator, tmp_path, [self.GOOD, self.GOOD])
        creator.plot_from_spec("plot altitude", schema)
        creator.plot_from_spec("make it red", schema)
        second = creator._spec_chain.invoke.call_args_list[1][0][0]
        assert "relative_alt" in second["history"]
        assert second["human_input"] == "make it red"

    def test_prompt_contains_units_and_format(self, creator, tmp_path):
        schema = self._ready(creator, tmp_path, [self.GOOD])
        creator.plot_from_spec("plot altitude", schema)
        sent = creator._spec_chain.invoke.call_args[0][0]
        assert '"unit": "m"' in sent["schema"]
        assert '"panels"' in sent["format"]

    def test_code_mode_still_available(self, clean_env):
        clean_env.setenv("OPENROUTER_API_KEY", "k")
        assert PlotCreator(mode="code").mode == "code"


class TestCliSpecOptions:

    def _spec_mode_creator(self, success):
        c = TestCliOutcome()._creator(success)
        c.mode = "spec"
        c.last_code = '{"panels": []}'
        c.spec_path = "/logs/plot.json"
        return c

    def test_spec_mode_success(self, capsys):
        from mavpose.chat.cli import run_once

        c = self._spec_mode_creator(True)
        assert run_once(c, "plot it") is True
        out = capsys.readouterr().out
        assert "Plot spec:" in out and "Spec saved to: /logs/plot.json" in out
        c.run_script.assert_not_called()

    def test_spec_mode_failure(self, capsys):
        from mavpose.chat.cli import run_once

        assert run_once(self._spec_mode_creator(False), "plot it") is False
        assert "couldn't be drawn" in capsys.readouterr().out

    def test_code_flag_selects_code_mode(self, clean_env, capsys, tmp_path):
        from mavpose.chat import cli

        clean_env.setenv("OPENROUTER_API_KEY", "k")
        seen = {}

        import mavpose.chat.plot_creator as pc

        class Spy(pc.PlotCreator):
            def __init__(self, **kw):
                seen.update(kw)
                super().__init__(**kw)

        clean_env.setattr(pc, "PlotCreator", Spy)
        clean_env.setattr(sys, "argv", ["mavpose", str(tmp_path / "nope.tlog"), "--code"])
        with pytest.raises(SystemExit):
            cli.main()
        assert seen["mode"] == "code"

    def test_redraw_saved_spec_without_any_model(self, clean_env, tmp_path, capsys):
        from mavpose.chat import cli
        from tests.log_builders import build_tlog

        log = build_tlog(tmp_path / "f.tlog")
        spec = tmp_path / "plot.json"
        spec.write_text(TestSpecMode.GOOD)
        clean_env.setattr(sys, "argv", ["mavpose", str(log), "--spec", str(spec)])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0          # no API key anywhere in clean_env
        assert (tmp_path / "plot.png").read_bytes()[:4] == b"\x89PNG"
        assert "Plot saved to" in capsys.readouterr().out

    def test_redraw_reports_a_spec_that_does_not_fit(self, clean_env, tmp_path, capsys):
        from mavpose.chat import cli
        from tests.log_builders import build_tlog

        log = build_tlog(tmp_path / "f.tlog")
        spec = tmp_path / "plot.json"
        spec.write_text(TestSpecMode.BAD)
        clean_env.setattr(sys, "argv", ["mavpose", str(log), "--spec", str(spec)])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert not (tmp_path / "plot.png").exists()
