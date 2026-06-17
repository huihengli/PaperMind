"""
PaperMind 论据抽取器测试
"""

import pytest

from src.argument_extractor import ArgumentExtractor


class TestArgumentExtractor:
    """论据抽取器单元测试"""

    def test_parse_llm_json_direct(self):
        extractor = ArgumentExtractor.__new__(ArgumentExtractor)
        content = '{"arguments": [{"id": "arg1", "statement": "Test", "type": "claim"}]}'
        result = extractor._parse_llm_json(content)
        assert len(result) == 1
        assert result[0]["statement"] == "Test"

    def test_parse_llm_json_markdown_block(self):
        extractor = ArgumentExtractor.__new__(ArgumentExtractor)
        content = '''```json
{"arguments": [{"id": "arg1", "statement": "Test in block", "type": "claim"}]}
```'''
        result = extractor._parse_llm_json(content)
        assert len(result) == 1
        assert result[0]["statement"] == "Test in block"

    def test_parse_llm_json_invalid(self):
        extractor = ArgumentExtractor.__new__(ArgumentExtractor)
        result = extractor._parse_llm_json("This is not JSON at all")
        assert result == []

    def test_validate_and_clean_empty(self):
        extractor = ArgumentExtractor.__new__(ArgumentExtractor)
        arguments = [{"id": "arg1", "statement": "", "type": "claim"}]
        result = extractor._validate_and_clean("paper_test", arguments)
        assert len(result) == 0

    def test_validate_and_clean_invalid_type(self):
        extractor = ArgumentExtractor.__new__(ArgumentExtractor)
        arguments = [{"id": "arg1", "statement": "Valid statement", "type": "invalid"}]
        result = extractor._validate_and_clean("paper_test", arguments)
        assert len(result) == 1
        assert result[0]["type"] == "claim"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
