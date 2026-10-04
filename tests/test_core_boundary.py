"""
tests/test_core_boundary.py

The core data layer must work without the chat assistant's dependencies,
and the chat layer must only use the core's public API.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import mavpose
from mavpose.chat import _deps

ROOT = Path(__file__).resolve().parent.parent
CHAT_DIR = ROOT / "mavpose" / "chat"
CORE_MODULES = ["mavpose", "mavpose.log_extractor", "mavpose.file_validator"]
CHAT_ONLY_DEPS = ["langchain_openai", "langchain_core", "langchain_chroma",
                  "chromadb", "openai", "tiktoken", "dotenv", "matplotlib"]


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=ROOT, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


class TestCoreIsIndependent:

    def test_importing_core_loads_no_chat_dependency(self):
        out = _run(
            "import sys, json\n"
            f"for m in {CORE_MODULES!r}: __import__(m)\n"
            f"print(json.dumps([m for m in {CHAT_ONLY_DEPS!r} if m in sys.modules]))\n"
        )
        assert json.loads(out) == []

    def test_core_extracts_a_log_without_chat_dependency(self, tmp_path):
        from tests.log_builders import build_tlog

        log = build_tlog(tmp_path / "flight.tlog")
        out = _run(
            "import sys, json\n"
            "from mavpose import LogExtractor\n"
            f"ex = LogExtractor({str(log)!r}); frames = ex.extract_all()\n"
            "alt = float(frames['GLOBAL_POSITION_INT']['alt'].max())\n"
            f"bad = [m for m in {CHAT_ONLY_DEPS!r} if m in sys.modules]\n"
            "print(json.dumps({'alt': alt, 'bad': bad}))\n"
        )
        result = json.loads(out)
        assert result["bad"] == []
        assert result["alt"] == pytest.approx(594.0)

    def test_core_never_imports_chat_package(self):
        for path in (ROOT / "mavpose").glob("*.py"):
            if path.name in {"PlotCreator.py", "safe_executor.py", "cli.py"}:
                continue  # deprecated re-export shims
            tree = ast.parse(path.read_text(encoding="utf-8"))
            top_level_imports = [
                node for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            for node in top_level_imports:
                names = [node.module] if isinstance(node, ast.ImportFrom) else [
                    a.name for a in node.names]
                assert not any((n or "").startswith("mavpose.chat") for n in names), (
                    f"{path.name} imports the chat package at module level")


class TestChatUsesPublicCoreApi:

    PRIVATE_CORE_ATTRS = re.compile(
        r"\._(frames|units|raw_units|schema|unknown_counts|extracted|"
        r"iter_messages|record_units)\b"
    )

    def test_chat_imports_only_public_core_names(self):
        for path in CHAT_DIR.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                        ("mavpose.log_extractor", "mavpose.file_validator")):
                    private = [a.name for a in node.names if a.name.startswith("_")]
                    assert not private, f"{path.name} imports private {private}"

    def test_chat_does_not_touch_extractor_internals(self):
        for path in CHAT_DIR.glob("*.py"):
            match = self.PRIVATE_CORE_ATTRS.search(path.read_text(encoding="utf-8"))
            assert match is None, f"{path.name} uses private core attribute {match.group()}"


class TestMissingChatExtra:

    def _pretend_missing(self, monkeypatch, *modules):
        real = _deps.importlib.util.find_spec
        monkeypatch.setattr(
            _deps.importlib.util, "find_spec",
            lambda name, *a, **k: None if name in modules else real(name, *a, **k),
        )

    def test_helpful_error(self, monkeypatch):
        self._pretend_missing(monkeypatch, "langchain_openai", "chromadb")
        with pytest.raises(_deps.ChatDependenciesMissing) as err:
            _deps.require_chat_deps()
        msg = str(err.value)
        assert "pip install 'mavpose[chat]'" in msg
        assert "langchain-openai" in msg and "chromadb" in msg
        assert isinstance(err.value, ImportError)

    def test_cli_exits_with_install_hint(self, monkeypatch, capsys, tmp_path):
        from mavpose.chat import cli

        monkeypatch.delitem(sys.modules, "mavpose.chat.plot_creator", raising=False)
        self._pretend_missing(monkeypatch, "langchain_openai")
        monkeypatch.setattr(sys, "argv", ["mavpose", str(tmp_path / "x.tlog"), "-p", "hi"])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == 1
        assert "pip install 'mavpose[chat]'" in capsys.readouterr().err

    def test_cli_help_works_without_extras(self):
        out = _run(
            "import sys\n"
            "sys.argv = ['mavpose', '--help']\n"
            "from mavpose.chat import cli\n"
            "try:\n    cli.main()\nexcept SystemExit:\n    pass\n"
            f"print([m for m in {CHAT_ONLY_DEPS!r} if m in sys.modules])\n"
        )
        assert "Path to a MAVLink log file" in out
        assert out.strip().endswith("[]")


class TestBackwardsCompatibility:

    def test_top_level_plotcreator_still_available(self):
        pytest.importorskip("langchain_openai", reason="needs the [chat] extra")
        from mavpose.chat.plot_creator import PlotCreator
        assert mavpose.PlotCreator is PlotCreator

    @pytest.mark.parametrize("old, new, name", [
        ("mavpose.safe_executor", "mavpose.chat.safe_executor", "execute_script"),
        ("mavpose.cli", "mavpose.chat.cli", "main"),
    ])
    def test_old_module_paths_warn_and_work(self, old, new, name):
        import importlib

        sys.modules.pop(old, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module = importlib.import_module(old)
        assert getattr(module, name) is getattr(importlib.import_module(new), name)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
