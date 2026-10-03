"""
tests/test_extract_code_snippets.py

Unit tests for PlotCreator.extract_code_snippets().
This is a pure static method with no LLM or file I/O — safe to test without mocking.
"""

from mavpose.PlotCreator import PlotCreator


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
