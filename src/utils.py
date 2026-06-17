"""
PaperMind 工具函数模块
提供数据序列化、日志、文本处理等通用工具
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import Config


# ============================================================
# 日志配置
# ============================================================

def setup_logger(name: str = "PaperMind") -> logging.Logger:
    """配置并返回 logger 实例"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_fmt)

        # 文件 handler
        log_dir = Config.BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"papermind_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        )
        file_handler.setFormatter(file_fmt)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# ============================================================
# ID 生成
# ============================================================

def generate_id(prefix: str = "") -> str:
    """生成唯一ID"""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{uid}" if prefix else uid


def generate_paper_id() -> str:
    return generate_id("paper")


def generate_argument_id() -> str:
    return generate_id("arg")


def generate_evidence_id() -> str:
    return generate_id("evd")


# ============================================================
# JSON 序列化
# ============================================================

class DateTimeEncoder(json.JSONEncoder):
    """支持 datetime 对象的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def save_json(data: Any, filepath: Path) -> None:
    """保存数据为 JSON 文件"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
    logger.info(f"JSON 已保存: {filepath}")


def load_json(filepath: Path) -> Optional[Dict]:
    """从 JSON 文件加载数据"""
    if not filepath.exists():
        logger.warning(f"文件不存在: {filepath}")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.debug(f"JSON 已加载: {filepath}")
    return data


# ============================================================
# 文本处理
# ============================================================

def clean_text(text: str) -> str:
    """清理多余空白和特殊字符"""
    import re
    # 移除多余换行（保留段落间的双换行）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 移除行内多余空格
    text = re.sub(r' {2,}', ' ', text)
    # 移除页码（常见模式）
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    return text.strip()


def truncate_text(text: str, max_length: int = 500) -> str:
    """截断文本，用于预览"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数量（中文 ~1.5 字符/token，英文 ~4 字符/token）"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


# ============================================================
# 论文元数据管理
# ============================================================

def get_paper_metadata_path(paper_id: str) -> Path:
    """获取论文字段数据的 JSON 路径"""
    return Config.PARSED_JSON_DIR / f"{paper_id}_metadata.json"


def get_argument_data_path(paper_id: str) -> Path:
    """获取论点数据的 JSON 路径"""
    return Config.PARSED_JSON_DIR / f"{paper_id}_arguments.json"


def get_compare_report_path(report_name: str) -> Path:
    """获取对比报告的存储路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Config.REPORTS_DIR / f"{report_name}_{timestamp}.md"


def list_indexed_papers() -> List[Dict[str, Any]]:
    """列出所有已成功解析和索引的论文（仅状态为 completed）"""
    papers = []
    if not Config.PARSED_JSON_DIR.exists():
        return papers

    for f in Config.PARSED_JSON_DIR.glob("*_metadata.json"):
        data = load_json(f)
        if data and data.get("status") == "completed":
            papers.append(data)
    # 按上传时间倒序
    papers.sort(key=lambda p: p.get("upload_time", ""), reverse=True)
    return papers


def delete_paper_data(paper_id: str) -> bool:
    """删除论文相关的所有数据"""
    import shutil

    deleted = False
    # 删除元数据
    meta_path = get_paper_metadata_path(paper_id)
    if meta_path.exists():
        meta_path.unlink()
        deleted = True

    # 删除论点数据
    arg_path = get_argument_data_path(paper_id)
    if arg_path.exists():
        arg_path.unlink()
        deleted = True

    # 删除原始 PDF
    pdf_dir = Config.RAW_PDF_DIR
    for pdf_file in pdf_dir.glob(f"{paper_id}_*"):
        pdf_file.unlink()
        deleted = True

    logger.info(f"论文数据已清理: {paper_id}")
    return deleted
