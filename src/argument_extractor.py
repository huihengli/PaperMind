"""
PaperMind 论点抽取模块
从论文文本中自动提取核心论点，结构化输出 JSON
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import Config, ArgumentType
from src.indexer import PaperIndexer
from src.utils import (
    estimate_tokens,
    generate_argument_id,
    get_argument_data_path,
    load_json,
    logger,
    save_json,
)


# ============================================================
# Prompt 模板
# ============================================================

EXTRACT_ARGUMENTS_SYSTEM = """你是一个专业的科研论文分析专家。你的任务是从论文内容中提取核心论点，并以结构化的 JSON 格式输出。

请严格遵循以下规则：
1. 提取 3-7 个核心论点
2. 每个论点用一句简洁的话概括（中文或英文均可，保持论文原始表述）
3. 标注论点类型：claim(主张)/hypothesis(假设)/conclusion(结论)
4. 尽量标注论点在原文中的大致位置（章节、段落）
5. 论点之间应互不重复，覆盖论文的主要贡献点"""

EXTRACT_ARGUMENTS_HUMAN = """论文标题：{title}

论文内容（节选）：
{context}

请从上述论文内容中提取核心论点。输出严格遵循以下 JSON 格式（不要包含其他文字）：

{{
  "arguments": [
    {{
      "id": "arg1",
      "statement": "<一句话概括论点>",
      "type": "claim|hypothesis|conclusion",
      "location": "<章节或段落位置，如 Section 3.2>"
    }}
  ]
}}"""


class ArgumentExtractor:
    """论点抽取器"""

    def __init__(self, indexer: PaperIndexer):
        """
        Args:
            indexer: PaperIndexer 实例，用于检索论文文本
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

    def extract_arguments(self, paper_id: str, paper_title: str) -> List[Dict[str, Any]]:
        """
        为指定论文抽取核心论点

        Args:
            paper_id: 论文ID
            paper_title: 论文标题

        Returns:
            论点列表
        """
        logger.info(f"开始论点抽取: {paper_title}")

        # 从向量数据库检索代表性文本块
        # 使用多个查询角度来覆盖论文的不同方面
        queries = [
            "核心创新点 主要贡献 方法",
            "实验设计 评估方法 结果分析",
            "结论 发现 局限性 未来工作",
            "假设 前提 理论依据",
        ]

        all_chunks = []
        seen_chunk_ids = set()
        for query in queries:
            chunks = self.indexer.search_similar(query, paper_id=paper_id, k=3)
            for chunk in chunks:
                chunk_id = chunk.metadata.get("chunk_id", "")
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_chunks.append(chunk)

        # 合并文本，控制 token 量
        context = "\n\n---\n\n".join([c.page_content for c in all_chunks])
        token_estimate = estimate_tokens(context)

        if token_estimate > 12000:
            # 如果太长，只取前部分
            logger.warning(f"文本过长 ({token_estimate} tokens)，将截断")
            context = context[:16000]  # 粗略截断

        logger.info(f"检索到 {len(all_chunks)} 个相关块 (~{token_estimate} tokens)")

        # 调用 LLM 抽取论点
        arguments = self._call_llm_extract(paper_id, paper_title, context)
        arguments = self._validate_and_clean(paper_id, arguments)

        # 保存结果
        self._save_arguments(paper_id, paper_title, arguments)

        logger.info(f"论点抽取完成: {len(arguments)} 个论点")
        return arguments

    def _call_llm_extract(
        self, paper_id: str, paper_title: str, context: str
    ) -> List[Dict[str, Any]]:
        """调用 LLM 进行论点抽取"""
        messages = [
            SystemMessage(content=EXTRACT_ARGUMENTS_SYSTEM),
            HumanMessage(content=EXTRACT_ARGUMENTS_HUMAN.format(
                title=paper_title,
                context=context,
            )),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content
            # 尝试解析 JSON
            arguments = self._parse_llm_json(content)
            return arguments
        except Exception as e:
            logger.error(f"LLM 论点抽取失败: {e}")
            return []

    def _parse_llm_json(self, content: str) -> List[Dict[str, Any]]:
        """解析 LLM 返回的 JSON"""
        # 尝试直接解析
        try:
            data = json.loads(content)
            return data.get("arguments", [])
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        json_match = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL
        )
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data.get("arguments", [])
            except json.JSONDecodeError:
                pass

        # 尝试从纯文本中找到 JSON 对象
        brace_match = re.search(r'\{.*"arguments".*\}', content, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return data.get("arguments", [])
            except json.JSONDecodeError:
                pass

        logger.warning(f"无法从 LLM 响应中解析 JSON: {content[:200]}...")
        return []

    def _validate_and_clean(
        self, paper_id: str, arguments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """验证并清理论点数据"""
        cleaned = []
        valid_types = {ArgumentType.CLAIM, ArgumentType.HYPOTHESIS, ArgumentType.CONCLUSION}

        for i, arg in enumerate(arguments):
            arg_id = arg.get("id") or generate_argument_id()
            statement = arg.get("statement", "").strip()
            arg_type = arg.get("type", "claim")

            # 跳过空论点
            if not statement:
                continue

            # 验证类型
            if arg_type not in valid_types:
                arg_type = ArgumentType.CLAIM

            cleaned.append({
                "id": arg.get("id", f"arg_{paper_id}_{i+1}"),
                "paper_id": paper_id,
                "statement": statement,
                "type": arg_type,
                "location": arg.get("location", "Unknown"),
            })

        # 限制最大论点数量
        if len(cleaned) > Config.MAX_ARGUMENTS_PER_PAPER:
            cleaned = cleaned[:Config.MAX_ARGUMENTS_PER_PAPER]

        return cleaned

    def _save_arguments(
        self, paper_id: str, paper_title: str, arguments: List[Dict[str, Any]]
    ):
        """保存论点数据到 JSON 文件"""
        data = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "argument_count": len(arguments),
            "arguments": arguments,
        }
        save_json(data, get_argument_data_path(paper_id))

    def load_arguments(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """加载已保存的论点数据"""
        path = get_argument_data_path(paper_id)
        return load_json(path)

    def get_argument_summaries(self, paper_id: str) -> str:
        """
        获取论文论点的文本摘要（用于后续 LLM 分析）

        Args:
            paper_id: 论文ID

        Returns:
            格式化的论点摘要文本
        """
        data = self.load_arguments(paper_id)
        if not data:
            return ""

        lines = [f"论文: {data.get('paper_title', 'Unknown')}"]
        for arg in data.get("arguments", []):
            lines.append(
                f"  - [{arg['type']}] {arg['statement']} "
                f"({arg.get('location', 'N/A')})"
            )
        return "\n".join(lines)
