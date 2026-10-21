"""
tests/test_extract_code_snippets.py

Unit tests for PlotCreator.extract_code_snippets().
This is a pure static method with no LLM or file I/O — safe to test without mocking.
"""

import pytest

pytest.importorskip("langchain_openai", reason="needs the [chat] extra")

from mavpose.chat.plot_creator import PlotCreator  # noqa: E402


class TestExtractCodeSnippets:
    """Tests for the extract_code_snippets static method."""

    def test_extracts_single_python_block(self):
        text = "Here is the script:\n```python\nprint('hello')\n```"
        result = PlotCreator.extract_code_snippets(text)
        assert result == ["print('hello')\n"]

    def test_extracts_multiple_blocks(self):
        text = "Block 1:\n```python\nx = 1\n```\nBlock 2:\n```python\ny = 2\n```"
        result = PlotCreator.extract_code_snippets(text)
        assert len(result) == 2
        assert "x = 1" in result[0]
        assert "y = 2" in result[1]

    def test_fallback_to_raw_text_when_no_fences(self):
        """If GPT returns raw code without fences, the whole text is returned."""
        raw = "import matplotlib\nprint('no fences')"
        result = PlotCreator.extract_code_snippets(raw)
        assert result == [raw]

    def test_empty_string_returns_list_with_empty_string(self):
        result = PlotCreator.extract_code_snippets("")
        assert result == [""]

    def test_block_with_no_language_tag(self):
        text = "Result:\n```\nsome code\n```"
        result = PlotCreator.extract_code_snippets(text)
        assert "some code" in result[0]

    def test_returns_list_type(self):
        result = PlotCreator.extract_code_snippets("```python\npass\n```")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_multiline_code_block(self):
        code = "import os\nfor i in range(10):\n    print(i)\n"
        text = f"```python\n{code}```"
        result = PlotCreator.extract_code_snippets(text)
        assert "import os" in result[0]
        assert "for i in range" in result[0]


class TestChoosingTheBlockToRun:
    """#13: the block that runs is the best Python block, never a shell snippet."""

    def test_bash_block_before_python_is_skipped(self):
        text = "First install:\n```bash\npip install pandas\n```\nThen:\n```python\nimport pandas\n```"
        assert PlotCreator.extract_code_snippets(text)[0] == "import pandas\n"

    def test_only_a_shell_block_is_never_returned_as_code(self):
        text = "```bash\nrm -rf /tmp/x\n```"
        assert PlotCreator.extract_code_snippets(text) == [text]

    def test_longest_python_block_wins(self):
        text = ("```python\nx = 1\n```\n"
                "```python\nimport pandas as pd\ndf = pd.read_parquet('t.parquet')\n```")
        assert PlotCreator.extract_code_snippets(text)[0].startswith("import pandas")

    def test_python_tag_beats_a_longer_unlabelled_block(self):
        text = "```\n" + "# notes\n" * 20 + "```\n```py\nprint(1)\n```"
        assert PlotCreator.extract_code_snippets(text)[0] == "print(1)\n"

    def test_unlabelled_block_used_when_no_python_tag(self):
        text = "```json\n{}\n```\n```\nprint(2)\n```"
        assert PlotCreator.extract_code_snippets(text) == ["print(2)\n"]

    def test_tag_with_trailing_space_and_case(self):
        assert PlotCreator.extract_code_snippets("```Python \nprint(3)\n```") == ["print(3)\n"]
