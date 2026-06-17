"""
PaperMind RAG 问答链
基于检索增强生成（RAG）实现论文问答功能
答案可追溯到原文具体位置
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from src.config import Config
from src.indexer import PaperIndexer
from src.utils import logger


# ============================================================
# Prompt 模板
# ============================================================

QA_SYSTEM_PROMPT = """你是一个科研论文问答助手。你将根据提供的论文内容回答用户的问题。

规则：
1. 严格基于提供的论文内容回答问题，不要引入外部知识
2. 如果论文内容不足以回答问题，请明确说明"根据论文内容，无法回答该问题"
3. 在回答中标注信息来源（章节、页码）
4. 尽量用简洁清晰的语言
5. 如果适用，可以引用原文片段"""

QA_HUMAN_TEMPLATE = """基于以下论文内容回答问题。

论文内容：
{context}

用户问题：{question}

请回答："""


class RAGChain:
    """RAG 问答链"""

    def __init__(self, indexer: PaperIndexer):
        """
        Args:
            indexer: PaperIndexer 实例
        """
        self.indexer = indexer
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE,
            max_retries=Config.LLM_MAX_RETRIES,
            request_timeout=Config.LLM_REQUEST_TIMEOUT,
        )

        # 对话记忆
        self._conversations: Dict[str, List[Dict[str, str]]] = {}

    def query(
        self,
        question: str,
        paper_id: Optional[str] = None,
        session_id: str = "default",
        k: int = None,
    ) -> Dict[str, Any]:
        """
        基于 RAG 回答用户问题

        Args:
            question: 用户问题
            paper_id: 可选，限定在特定论文内
            session_id: 会话ID（用于多轮对话）
            k: 检索文档数量

        Returns:
            {
                "answer": "回答文本",
                "sources": [{"text": "原文片段", "section": "章节", "page": 页码}],
                "question": "原始问题"
            }
        """
        logger.info(f"RAG 问答: [{session_id}] {question[:80]}...")

        # 检索相关文档块
        if k is None:
            k = Config.RETRIEVER_K

        docs = self.indexer.search_similar(question, paper_id=paper_id, k=k)

        if not docs:
            return {
                "answer": "未在论文中找到相关信息，请确认论文已正确索引。",
                "sources": [],
                "question": question,
            }

        # 构造上下文
        context_parts = []
        sources = []
        for doc in docs:
            meta = doc.metadata
            context_parts.append(
                f"[来源: {meta.get('section', 'Unknown')}, "
                f"Page {meta.get('page', '?')}]\n{doc.page_content}"
            )
            sources.append({
                "text": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                "section": meta.get("section", "Unknown"),
                "page": meta.get("page", "?"),
                "paper_title": meta.get("paper_title", "Unknown"),
                "chunk_id": meta.get("chunk_id", ""),
            })

        context = "\n\n---\n\n".join(context_parts)

        # 加载对话历史
        history = self._get_history(session_id)

        # 构造 messages
        messages = [SystemMessage(content=QA_SYSTEM_PROMPT)]

        # 添加历史对话
        for turn in history[-4:]:  # 只保留最近 4 轮
            messages.append(HumanMessage(content=turn["question"]))
            messages.append(SystemMessage(content=turn["answer"]))

        # 添加当前问题
        messages.append(HumanMessage(content=QA_HUMAN_TEMPLATE.format(
            context=context,
            question=question,
        )))

        # 调用 LLM
        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            logger.error(f"RAG 问答 LLM 调用失败: {e}")
            answer = f"抱歉，处理您的问题时出现错误: {e}"

        # 保存对话历史
        self._save_turn(session_id, question, answer)

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "paper_id": paper_id,
        }

    def _get_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取会话历史"""
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        return self._conversations[session_id]

    def _save_turn(self, session_id: str, question: str, answer: str):
        """保存一轮对话"""
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        self._conversations[session_id].append({
            "question": question,
            "answer": answer,
        })

    def clear_history(self, session_id: str = "default"):
        """清除会话历史"""
        if session_id in self._conversations:
            del self._conversations[session_id]
            logger.info(f"已清除会话历史: {session_id}")

    def get_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """获取会话历史"""
        return self._get_history(session_id)

    def get_relevant_chunks(
        self, question: str, paper_id: Optional[str] = None, k: int = 5
    ) -> List[Document]:
        """获取与问题相关的文档块（不含 LLM 回答）"""
        return self.indexer.search_similar(question, paper_id=paper_id, k=k)
