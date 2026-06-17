"""
PaperMind PDF 加载器测试
"""

import pytest
from pathlib import Path

from src.pdf_loader import PDFLoader


class TestPDFLoader:
    """PDF 加载器单元测试"""

    def test_init(self):
        loader = PDFLoader()
        assert loader is not None

    def test_load_nonexistent_pdf(self):
        loader = PDFLoader()
        fake_path = Path("/nonexistent/file.pdf")
        # 应该抛出异常
        with pytest.raises(Exception):
            loader.load_pdf(fake_path)

    def test_extract_metadata_empty(self):
        loader = PDFLoader()
        metadata = loader._extract_metadata(Path("fake.pdf"), "")
        assert "title" in metadata
        assert "page_count" in metadata
        assert metadata["page_count"] == 0

    def test_clean_text_basic(self):
        from src.utils import clean_text
        dirty = "Hello   World\n\n\n\n\nGoodbye"
        clean = clean_text(dirty)
        assert "  " not in clean
        assert "\n\n\n" not in clean


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
