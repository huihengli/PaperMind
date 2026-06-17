"""
PaperMind 配置管理模块
负责加载环境变量、管理模型配置、定义常量
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Config:
    """全局配置类"""

    # ---- 路径配置 ----
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
    PARSED_JSON_DIR = DATA_DIR / "parsed_json"
    REPORTS_DIR = DATA_DIR / "reports"

    # ---- OpenAI 配置 ----
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # ---- Chroma 配置 ----
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "papermind_papers")

    # ---- 分块配置 ----
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ---- 应用配置 ----
    MAX_COMPARE_PAPERS = int(os.getenv("MAX_COMPARE_PAPERS", "5"))
    MAX_ARGUMENTS_PER_PAPER = int(os.getenv("MAX_ARGUMENTS_PER_PAPER", "7"))

    # ---- LLM 调用配置 ----
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))

    # ---- 检索配置 ----
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", "8"))

    @classmethod
    def validate(cls) -> bool:
        """验证必要的配置项"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY 未设置。请在 .env 文件中配置，"
                "或设置环境变量 OPENAI_API_KEY"
            )
        return True

    @classmethod
    def ensure_dirs(cls):
        """确保所有必要的目录存在"""
        for d in [cls.DATA_DIR, cls.RAW_PDF_DIR, cls.PARSED_JSON_DIR, cls.REPORTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        Path(cls.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)


# 论文状态枚举
class PaperStatus:
    UPLOADED = "uploaded"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


# 论点类型枚举
class ArgumentType:
    CLAIM = "claim"           # 主张
    HYPOTHESIS = "hypothesis" # 假设
    CONCLUSION = "conclusion" # 结论


# 关系类型枚举
class RelationType:
    SUPPORTS = "supports"         # A 支持 B
    CONTRADICTS = "contradicts"   # A 与 B 矛盾
    ELABORATES = "elaborates"     # A 细化为 B
    LEADS_TO = "leads_to"         # A 导致 B（因果）


# 关系类型中文映射
RELATION_TYPE_CN = {
    RelationType.SUPPORTS: "支持",
    RelationType.CONTRADICTS: "矛盾",
    RelationType.ELABORATES: "细化",
    RelationType.LEADS_TO: "因果",
}

# 关系类型颜色映射（用于可视化）
RELATION_TYPE_COLORS = {
    RelationType.SUPPORTS: "#4CAF50",     # 绿色
    RelationType.CONTRADICTS: "#F44336",  # 红色
    RelationType.ELABORATES: "#2196F3",   # 蓝色
    RelationType.LEADS_TO: "#FF9800",     # 橙色
}
