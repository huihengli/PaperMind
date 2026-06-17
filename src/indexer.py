"""
PaperMind 索引构建模块
负责文本分块、向量化、存入 Chroma 向量数据库
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import Config, PaperStatus
from src.utils import (
    generate_id,
    get_paper_metadata_path,
    load_json,
    logger,
    save_json,
)


class PaperIndexer:
    """论文索引构建器"""

    def __init__(self):
        Config.validate()
        Config.ensure_dirs()

        # 根据 API Base 自动选择 Embedding 提供商
        # 非 OpenAI 官方 API（如 DeepSeek）不支持 embedding，使用本地 HuggingFace 模型
        self.embeddings = self._create_embeddings()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "。", " ", ""],
            length_function=len,
        )

        self.vectorstore: Optional[Chroma] = None
        self._init_vectorstore()

    def _create_embeddings(self):
        """根据配置自动选择 Embedding 提供商"""
        api_base = (Config.OPENAI_API_BASE or "").lower()

        # 如果是标准 OpenAI API，使用 OpenAI Embedding
        if "api.openai.com" in api_base or not api_base:
            logger.info(f"使用 OpenAI Embedding: {Config.EMBEDDING_MODEL}")
            return OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY,
                openai_api_base=Config.OPENAI_API_BASE or None,
            )

        # 否则（如 DeepSeek 等第三方 API）使用本地 HuggingFace Embedding
        logger.info("检测到非 OpenAI API，使用本地 HuggingFace Embedding (all-MiniLM-L6-v2)")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _init_vectorstore(self):
        """初始化向量存储"""
        persist_dir = Config.CHROMA_PERSIST_DIR
        self.vectorstore = Chroma(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir),
        )
        logger.info(f"Chroma 向量存储已初始化: {persist_dir}")

    def index_paper(self, paper_id: str, full_text: str) -> int:
        """
        为单篇论文构建索引

        Args:
            paper_id: 论文ID
            full_text: 论文全文

        Returns:
            创建的文档块数量
        """
        logger.info(f"开始为论文构建索引: {paper_id}")

        # 获取论文元数据
        meta_path = get_paper_metadata_path(paper_id)
        paper_data = load_json(meta_path) if meta_path.exists() else {}

        # 更新状态为 indexing
        if paper_data:
            paper_data["status"] = PaperStatus.INDEXING
            save_json(paper_data, meta_path)

        # 文本分块
        documents = self._split_text(full_text, paper_id, paper_data)

        # 存入向量数据库
        try:
            self.vectorstore.add_documents(documents)
            logger.info(f"成功索引 {len(documents)} 个文本块: {paper_id}")
        except Exception as e:
            logger.error(f"向量存储失败: {e}")
            if paper_data:
                paper_data["status"] = PaperStatus.FAILED
                paper_data["error"] = str(e)
                save_json(paper_data, meta_path)
            raise

        # 更新状态为 completed
        if paper_data:
            paper_data["status"] = PaperStatus.COMPLETED
            paper_data["indexed_chunks"] = len(documents)
            paper_data["indexed_time"] = datetime.now()
            save_json(paper_data, meta_path)

        return len(documents)

    def delete_paper(self, paper_id: str) -> int:
        """
        从向量数据库中删除指定论文的所有文档块

        Args:
            paper_id: 论文ID

        Returns:
            删除的文档块数量
        """
        logger.info(f"从向量库删除论文: {paper_id}")
        try:
            # 通过 Chroma collection 的 metadata 过滤删除
            collection = self.vectorstore._collection
            result = collection.get(where={"paper_id": paper_id})
            ids = result.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                logger.info(f"已从向量库删除 {len(ids)} 个块: {paper_id}")
            return len(ids)
        except Exception as e:
            logger.error(f"向量库删除失败 [{paper_id}]: {e}")
            return 0

    def index_papers_batch(
        self,
        texts: Dict[str, str],
        progress_callback=None
    ) -> Dict[str, int]:
        """
        批量索引多篇论文

        Args:
            texts: {paper_id: full_text} 的字典
            progress_callback: 可选的进度回调

        Returns:
            {paper_id: chunk_count} 的结果字典
        """
        results = {}
        total = len(texts)

        for i, (paper_id, full_text) in enumerate(texts.items()):
            try:
                chunk_count = self.index_paper(paper_id, full_text)
                results[paper_id] = chunk_count
            except Exception as e:
                logger.error(f"论文索引失败 [{paper_id}]: {e}")
                results[paper_id] = -1

            if progress_callback:
                progress_callback((i + 1) / total)

        return results

    def _split_text(
        self,
        full_text: str,
        paper_id: str,
        paper_data: Dict[str, Any]
    ) -> List[Document]:
        """
        将论文全文切分为文档块

        Args:
            full_text: 论文全文
            paper_id: 论文ID
            paper_data: 论文元数据

        Returns:
            LangChain Document 列表
        """
        # 使用 LangChain 的分块器
        chunks = self.text_splitter.split_text(full_text)

        documents = []
        for i, chunk_text in enumerate(chunks):
            # 尝试推断章节
            section = self._infer_section(chunk_text, paper_data)

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "paper_id": paper_id,
                    "paper_title": paper_data.get("title", "Unknown"),
                    "section": section,
                    "page": self._infer_page(i, len(chunks), paper_data),
                    "chunk_index": i,
                    "chunk_id": generate_id("chunk"),
                    "total_chunks": len(chunks),
                },
            )
            documents.append(doc)

        logger.info(f"文本已切分为 {len(documents)} 个块 (size={Config.CHUNK_SIZE}, "
                      f"overlap={Config.CHUNK_OVERLAP})")
        return documents

    def _infer_section(self, chunk_text: str, paper_data: Dict[str, Any]) -> str:
        """根据文本内容推断所属章节"""
        sections = paper_data.get("sections", [])
        for sec in sections:
            sec_text = sec.get("text", "")
            if chunk_text[:50] in sec_text or chunk_text[-50:] in sec_text:
                return sec.get("name", "Unknown")
        return "Unknown"

    def _infer_page(
        self, chunk_index: int, total_chunks: int, paper_data: Dict[str, Any]
    ) -> int:
        """根据块索引推断页码"""
        page_count = paper_data.get("page_count", 1)
        return int(chunk_index / total_chunks * page_count) + 1

    def search_similar(
        self,
        query: str,
        paper_id: Optional[str] = None,
        k: int = None
    ) -> List[Document]:
        """
        检索与查询最相关的文档块

        Args:
            query: 查询文本
            paper_id: 可选，限定在特定论文内检索
            k: 返回结果数量

        Returns:
            相关文档块列表
        """
        if k is None:
            k = Config.RETRIEVER_K

        if paper_id:
            # 带过滤条件的检索
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"paper_id": paper_id},
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)

        logger.debug(f"检索返回 {len(results)} 个结果 [query={query[:50]}...]")
        return results

    def search_with_score(
        self,
        query: str,
        paper_id: Optional[str] = None,
        k: int = None
    ) -> List[tuple]:
        """检索并返回相似度分数"""
        if k is None:
            k = Config.RETRIEVER_K

        if paper_id:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter={"paper_id": paper_id},
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

    def delete_paper_index(self, paper_id: str) -> bool:
        """从向量数据库中删除指定论文的所有块"""
        try:
            # Chroma 支持按元数据过滤删除
            collection = self.vectorstore._collection
            collection.delete(where={"paper_id": paper_id})
            logger.info(f"已从索引中删除论文: {paper_id}")
            return True
        except Exception as e:
            logger.error(f"删除索引失败 [{paper_id}]: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取向量数据库统计信息"""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "total_chunks": count,
                "collection_name": Config.CHROMA_COLLECTION_NAME,
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
