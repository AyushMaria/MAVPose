"""
tests/test_safe_executor.py

Unit tests for mavpose/chat/safe_executor.py (stdlib + pandas only, so
these also run in a core-only install).
Verifies that the subprocess sandbox correctly allows safe code
and blocks dangerous imports and operations.
"""

import pytest

from mavpose.chat.safe_executor import execute_script


class TestExecuteScript:

    def test_simple_arithmetic(self):
        success, output = execute_script("result = 1 + 1")
        assert success

    def test_allowed_import_math(self):
        success, output = execute_script("import math\nresult = math.sqrt(4)")
        assert success

    def test_allowed_import_json(self):
        success, output = execute_script("import json\ndata = json.dumps({'key': 'value'})")
        assert success

    def test_blocks_os_import(self):
        success, output = execute_script("import os\nos.system('echo pwned')")
        assert not success
        assert "not allowed" in output or "ImportError" in output

    def test_blocks_subprocess_import(self):
        success, output = execute_script("import subprocess\nsubprocess.run(['ls'])")
        assert not success

    def test_blocks_sys_import(self):
        success, output = execute_script("import sys\nsys.exit(0)")
        assert not success

    def test_syntax_error_returns_failure(self):
        success, output = execute_script("def broken(\n    pass")
        assert not success
        assert "SyntaxError" in output

    def test_timeout_enforced(self):
        # Infinite loop should time out
        success, output = execute_script("while True: pass", timeout_seconds=2)
        assert not success
        assert "timed out" in output.lower()

    def test_returns_tuple(self):
        result = execute_script("x = 1")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestSandboxHardening:
    """Regression tests for escapes found in the October 2026 review."""

    def test_real_import_not_leaked_into_globals(self):
        code = (
            "import builtins\n"
            "builtins.__import__ = _real_import\n"
            "import os\n"
            "os.system('echo ' + 'ESC' + 'APED')\n"
        )
        success, output = execute_script(code)
        assert not success
        assert "ESCAPED" not in output

    def test_os_system_via_allowed_library_blocked(self):
        code = "import pandas as pd\npd.io.common.os.system('echo ' + 'ESC' + 'APED')\n"
        success, output = execute_script(code)
        assert not success
        assert "ESCAPED" not in output
        assert "Blocked by MAVPose sandbox" in output

    def test_subprocess_via_allowed_library_blocked(self):
        code = "import pandas as pd\npd.io.common.os.popen('echo ' + 'ESC' + 'APED').read()\n"
        success, output = execute_script(code)
        assert not success
        assert "subprocess.Popen" in output

    def test_write_outside_allowed_dir_blocked(self, tmp_path):
        target = tmp_path / "outside" / "pwn.txt"
        target.parent.mkdir()
        allowed = tmp_path / "out"
        allowed.mkdir()
        success, output = execute_script(
            f"open({str(target)!r}, 'w').write('x')\n",
            allowed_write_dirs=[str(allowed)],
        )
        assert not success
        assert not target.exists()

    def test_write_inside_allowed_dir_permitted(self, tmp_path):
        target = tmp_path / "plot.txt"
        success, output = execute_script(
            f"open({str(target)!r}, 'w').write('ok')\n",
            allowed_write_dirs=[str(tmp_path)],
        )
        assert success, output
        assert target.read_text() == "ok"

    def test_delete_outside_allowed_dir_blocked(self, tmp_path):
        victim = tmp_path / "keep.txt"
        victim.write_text("important")
        code = f"import pathlib\npathlib.Path({str(victim)!r}).unlink()\n"
        success, output = execute_script(code, allowed_write_dirs=[str(tmp_path / "out")])
        assert not success
        assert victim.exists()

    def test_network_blocked(self):
        code = "import pandas.io.common as c\nc.urlopen('http://example.com')\n"
        success, output = execute_script(code)
        assert not success
        assert "Blocked by MAVPose sandbox" in output

    def test_api_key_not_visible(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-must-not-leak")
        code = (
            "import pandas as pd\n"
            "print('KEY=' + str(pd.io.common.os.environ.get('OPENROUTER_API_KEY')))\n"
        )
        success, output = execute_script(code)
        assert success, output
        assert "sk-must-not-leak" not in output

    def test_traceback_reports_script_line_numbers(self):
        success, output = execute_script("x = 1\ny = 2\nraise ValueError('boom')\n")
        assert not success
        assert 'plot_script.py", line 3' in output

    def test_matplotlib_plot_still_works(self, tmp_path):
        pytest.importorskip("matplotlib", reason="needs the [chat] extra")
        png = tmp_path / "plot.png"
        code = (
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "df = pd.DataFrame({'t': [0, 1, 2], 'v': [1, 4, 9]})\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(df['t'], df['v'])\n"
            f"fig.savefig({str(png)!r})\n"
        )
        success, output = execute_script(code, allowed_write_dirs=[str(tmp_path)])
        assert success, output
        assert png.exists()

    def test_ctypes_via_library_blocked_after_import(self):
        code = "import numpy.ctypeslib\nnumpy.ctypeslib.ctypes.CDLL(None)\n"
        success, output = execute_script(code)
        assert not success
        assert "Blocked by MAVPose sandbox: ctypes." in output

    def test_cannot_import_module_written_by_script(self, tmp_path):
        code = (
            f"open({str(tmp_path / 'evil.py')!r}, 'w').write('X = 1')\n"
            "import inspect\n"
            f"inspect.sys.path.insert(0, {str(tmp_path)!r})\n"
            "import evil\n"
        )
        success, output = execute_script(code, allowed_write_dirs=[str(tmp_path)])
        assert not success
        assert "Blocked by MAVPose sandbox: compile" in output
