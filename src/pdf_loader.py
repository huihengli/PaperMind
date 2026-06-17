"""
PaperMind PDF 加载与解析模块
支持 PDF 文本提取、章节识别、元数据抽取
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber
from PyPDF2 import PdfReader

from src.config import Config, PaperStatus
from src.utils import (
    clean_text,
    generate_paper_id,
    get_paper_metadata_path,
    load_json,
    logger,
    save_json,
)


class PDFLoader:
    """PDF 论文加载器，负责解析和提取文本"""

    def __init__(self):
        Config.ensure_dirs()

    def load_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        加载单个 PDF 文件并提取全文

        Args:
            file_path: PDF 文件路径

        Returns:
            包含论文元数据和全文的字典
        """
        logger.info(f"开始解析 PDF: {file_path.name}")

        paper_id = generate_paper_id()

        # 复制 PDF 到数据目录
        saved_path = self._save_pdf(file_path, paper_id)

        # 提取文本
        full_text = self._extract_text(saved_path)

        # 提取章节结构
        sections = self._extract_sections(full_text)

        # 提取元数据
        metadata = self._extract_metadata(saved_path, full_text)

        # 构建论文对象
        paper = {
            "id": paper_id,
            "title": metadata.get("title", file_path.stem),
            "filename": file_path.name,
            "file_path": str(saved_path),
            "upload_time": datetime.now(),
            "page_count": metadata.get("page_count", 0),
            "status": PaperStatus.UPLOADED,
            "sections": sections,
            "full_text_length": len(full_text),
            "metadata": metadata,
        }

        # 保存元数据
        save_json(paper, get_paper_metadata_path(paper_id))

        logger.info(f"PDF 解析完成: {paper['title']} (pages={paper['page_count']})")
        return paper

    def load_pdfs_batch(
        self, file_paths: List[Path], progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        批量加载多个 PDF 文件

        Args:
            file_paths: PDF 文件路径列表
            progress_callback: 可选的进度回调函数

        Returns:
            论文对象列表
        """
        papers = []
        total = len(file_paths)

        for i, fp in enumerate(file_paths):
            try:
                paper = self.load_pdf(fp)
                papers.append(paper)
            except Exception as e:
                logger.error(f"解析 PDF 失败 [{fp.name}]: {e}")
                # 即使失败也记录一条记录
                papers.append({
                    "id": generate_paper_id(),
                    "title": fp.stem,
                    "filename": fp.name,
                    "file_path": str(fp),
                    "upload_time": datetime.now(),
                    "status": PaperStatus.FAILED,
                    "error": str(e),
                })

            if progress_callback:
                progress_callback((i + 1) / total)

        return papers

    def _save_pdf(self, source_path: Path, paper_id: str) -> Path:
        """将 PDF 复制到数据目录"""
        dest_dir = Config.RAW_PDF_DIR
        dest_path = dest_dir / f"{paper_id}_{source_path.name}"
        shutil.copy2(source_path, dest_path)
        return dest_path

    def _extract_text(self, pdf_path: Path) -> str:
        """
        从 PDF 中提取纯文本
        优先使用 pdfplumber（效果好），
        降级到 PyPDF2（兼容性广）
        """
        full_text_parts = []

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text_parts.append(text)
        except Exception as e:
            logger.warning(f"pdfplumber 解析失败，降级到 PyPDF2: {e}")
            full_text_parts = self._extract_with_pypdf2(pdf_path)

        full_text = "\n\n".join(full_text_parts)
        return clean_text(full_text)

    def _extract_with_pypdf2(self, pdf_path: Path) -> List[str]:
        """PyPDF2 降级方案"""
        text_parts = []
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return text_parts

    def _extract_metadata(self, pdf_path: Path, full_text: str) -> Dict[str, Any]:
        """从 PDF 和文本中提取元数据"""
        metadata: Dict[str, Any] = {
            "title": "",
            "authors": [],
            "page_count": 0,
        }

        try:
            # 使用 pdfplumber 获取页数
            with pdfplumber.open(str(pdf_path)) as pdf:
                metadata["page_count"] = len(pdf.pages)
        except Exception:
            try:
                reader = PdfReader(str(pdf_path))
                metadata["page_count"] = len(reader.pages)
            except Exception:
                pass

        # 尝试从文本开头提取标题（通常论文第一行是标题）
        first_lines = full_text.split("\n")[:5]
        if first_lines:
            # 过滤掉明显不是标题的行（太短、纯数字等）
            for line in first_lines:
                line = line.strip()
                if len(line) > 20 and not line.isdigit():
                    metadata["title"] = line
                    break

        return metadata

    def _extract_sections(self, full_text: str) -> List[Dict[str, Any]]:
        """
        尝试按章节切分文本，识别常见章节标题
        """
        import re

        # 常见章节标题模式
        section_patterns = [
            r'(?:^|\n)\s*(?:Abstract|摘要)\s*\n',
            r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Introduction|引言|介绍)\s*\n',
            r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Related\s*Work|相关工作|文献综述)\s*\n',
            r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Method|Methodology|Approach|方法|模型)\s*\n',
            r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Experiments?|Evaluation|实验|评估)\s*\n',
            r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Results?|结果)\s*\n',
            r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Discussion|讨论|分析)\s*\n',
            r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:Conclusion|总结|结论)\s*\n',
        ]

        sections = []
        text = full_text

        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                section_name = matches[0].group().strip()
                start = matches[0].start()
                sections.append({
                    "name": section_name,
                    "start": start,
                })

        # 计算每个章节的文本范围
        for i, sec in enumerate(sections):
            sec_start = sec["start"]
            if i + 1 < len(sections):
                sec_end = sections[i + 1]["start"]
            else:
                sec_end = len(text)
            sec["text"] = text[sec_start:sec_end].strip()
            sec["length"] = len(sec["text"])

        return sections

    def get_full_text(self, paper_id: str) -> Optional[str]:
        """
        获取指定论文的完整文本

        Args:
            paper_id: 论文ID

        Returns:
            论文全文文本，如果论文不存在则返回 None
        """
        meta_path = get_paper_metadata_path(paper_id)
        paper_data = load_json(meta_path) if meta_path.exists() else None
        if not paper_data:
            return None

        pdf_path = Path(paper_data["file_path"])
        if not pdf_path.exists():
            return None

        return self._extract_text(pdf_path)
