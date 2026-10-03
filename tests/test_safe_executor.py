"""
tests/test_safe_executor.py

Unit tests for mavpose/safe_executor.py.
Verifies that the subprocess sandbox correctly allows safe code
and blocks dangerous imports and operations.
"""

from mavpose.safe_executor import execute_script


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
