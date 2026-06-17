"""
PaperMind 关系识别模块
分析论点之间的逻辑关系：支持、矛盾、细化、因果
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import Config, RelationType
from src.utils import (
    generate_argument_id,
    get_argument_data_path,
    load_json,
    logger,
    save_json,
)


# ============================================================
# Prompt 模板
# ============================================================

IDENTIFY_RELATIONS_SYSTEM = """你是一个逻辑分析专家。你需要分析科研论文论点之间的逻辑关系。

关系类型定义：
- supports: 论点A是论点B的证据或理由（A 支持 B）
- contradicts: 论点A与论点B矛盾或冲突
- elaborates: 论点B是论点A的具体展开或细化（A 细化为 B）
- leads_to: 论点A导致论点B（因果关系，A → B）

注意：
1. 只输出有意义的关系，不要强行关联无关的论点
2. 如果两两之间没有明确关系，不要输出
3. 确保每个关系都有合理的逻辑依据"""

IDENTIFY_RELATIONS_HUMAN = """论文标题: {title}

论点列表:
{arguments}

请分析以上论点之间的逻辑关系。输出格式（严格 JSON）：

{{
  "relations": [
    {{"from": "arg1", "to": "arg2", "type": "supports", "explanation": "理由简述"}}
  ]
}}"""


class RelationIdentifier:
    """关系识别器"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.2,  # 更低温度以获得一致性
            max_tokens=Config.LLM_MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE,
            max_retries=Config.LLM_MAX_RETRIES,
            request_timeout=Config.LLM_REQUEST_TIMEOUT,
        )

    def identify_relations(
        self,
        paper_id: str,
        paper_title: str,
        arguments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        识别论点之间的关系

        Args:
            paper_id: 论文ID
            paper_title: 论文标题
            arguments: 论点列表

        Returns:
            关系列表
        """
        logger.info(f"开始关系识别: {paper_title} (arguments={len(arguments)})")

        if len(arguments) < 2:
            logger.info("论点数量不足，跳过关系识别")
            return []

        # 构造论点摘要
        arg_lines = []
        for arg in arguments:
            arg_lines.append(
                f"  [{arg.get('id', '?')}] ({arg.get('type', 'claim')}) "
                f"{arg.get('statement', '')}"
            )
        arg_text = "\n".join(arg_lines)

        # 调用 LLM
        relations = self._call_llm_identify(paper_title, arg_text)

        # 验证和清理
        relations = self._validate_relations(arguments, relations)

        # 保存结果
        self._save_relations(paper_id, paper_title, arguments, relations)

        logger.info(f"关系识别完成: {len(relations)} 个关系")
        return relations

    def _call_llm_identify(
        self, paper_title: str, arg_text: str
    ) -> List[Dict[str, Any]]:
        """调用 LLM 进行关系识别"""
        messages = [
            SystemMessage(content=IDENTIFY_RELATIONS_SYSTEM),
            HumanMessage(content=IDENTIFY_RELATIONS_HUMAN.format(
                title=paper_title,
                arguments=arg_text,
            )),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content
            return self._parse_llm_json(content)
        except Exception as e:
            logger.error(f"LLM 关系识别失败: {e}")
            return []

    def _parse_llm_json(self, content: str) -> List[Dict[str, Any]]:
        """解析 LLM 返回的关系 JSON"""
        # 尝试直接解析
        try:
            data = json.loads(content)
            return data.get("relations", [])
        except json.JSONDecodeError:
            pass

        # 从 markdown 代码块中提取
        json_match = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL
        )
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data.get("relations", [])
            except json.JSONDecodeError:
                pass

        # 从纯文本中找到 JSON 对象
        brace_match = re.search(r'\{.*"relations".*\}', content, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return data.get("relations", [])
            except json.JSONDecodeError:
                pass

        logger.warning(f"无法从 LLM 响应中解析关系 JSON: {content[:200]}...")
        return []

    def _validate_relations(
        self,
        arguments: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """验证关系数据的有效性"""
        valid_types = {
            RelationType.SUPPORTS,
            RelationType.CONTRADICTS,
            RelationType.ELABORATES,
            RelationType.LEADS_TO,
        }

        # 收集所有有效的论点 ID
        valid_arg_ids = {arg["id"] for arg in arguments}

        cleaned = []
        for rel in relations:
            from_id = rel.get("from", "")
            to_id = rel.get("to", "")
            rel_type = rel.get("type", "")

            # 跳过无效关系
            if from_id not in valid_arg_ids or to_id not in valid_arg_ids:
                continue
            if rel_type not in valid_types:
                continue
            if from_id == to_id:
                continue

            cleaned.append({
                "from": from_id,
                "to": to_id,
                "type": rel_type,
                "explanation": rel.get("explanation", ""),
            })

        return cleaned

    def _save_relations(
        self,
        paper_id: str,
        paper_title: str,
        arguments: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ):
        """保存关系数据到论点的 JSON 文件中"""
        # 读取已有的论点数据
        arg_path = get_argument_data_path(paper_id)
        data = load_json(arg_path) or {}

        data["paper_id"] = paper_id
        data["paper_title"] = paper_title
        data["arguments"] = arguments
        data["relations"] = relations
        data["relation_count"] = len(relations)

        save_json(data, arg_path)

    def load_relations(
        self, paper_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        加载已保存的论点和关系数据

        Returns:
            (arguments, relations) 元组
        """
        data = load_json(get_argument_data_path(paper_id))
        if not data:
            return [], []
        return data.get("arguments", []), data.get("relations", [])
