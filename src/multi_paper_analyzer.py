"""
PaperMind 多篇论文对比分析模块
实现跨论文的核心论点对比、局限性归纳、研究空白识别和方向建议
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import Config
from src.argument_extractor import ArgumentExtractor
from src.indexer import PaperIndexer
from src.utils import (
    estimate_tokens,
    get_argument_data_path,
    get_compare_report_path,
    load_json,
    logger,
    save_json,
)


# ============================================================
# Prompt 模板
# ============================================================

COMPARE_SYSTEM = """你是一个科研论文对比分析专家。你需要对多篇论文进行系统性的对比分析。

分析要点：
1. 提炼每篇论文的核心方法和创新点
2. 识别多篇论文之间的共性和差异
3. 找出观点矛盾或方法冲突的地方
4. 基于各论文的局限性，推理可能的研究空白
5. 提出合理可行的未来研究方向"""

COMPARE_HUMAN = """请对比分析以下 {num_papers} 篇论文：

{paper_summaries}

请输出一份包含以下部分的详细分析报告（Markdown 格式）：

## 1. 核心论点对比

| 论文 | 核心方法 | 关键创新点 | 主要局限性 |
|------|---------|-----------|-----------|
{paper_rows}

## 2. 共同点归纳

- 所有论文都认同的核心观点
- 共同采用或遵循的方法论
- 一致的研究趋势

## 3. 差异点分析

- 方法论层面的差异
- 假设前提的不同
- 评估标准和数据集差异
- 矛盾或冲突的观点

## 4. 研究空白识别

- 各论文均未覆盖的领域
- 共同局限性的深层原因
- 潜在的研究机会

## 5. 未来研究方向建议

给出 2-4 个具体可行的研究方向，每个方向包含：
- 方向名称
- 研究动机（为什么值得做）
- 大致方法思路
- 预期贡献"""


class MultiPaperAnalyzer:
    """多篇论文对比分析器"""

    def __init__(self, indexer: PaperIndexer, argument_extractor: ArgumentExtractor):
        """
        Args:
            indexer: PaperIndexer 实例
            argument_extractor: ArgumentExtractor 实例
        """
        self.indexer = indexer
        self.argument_extractor = argument_extractor

        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=3000,  # 对比分析需要更多 token
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE,
            max_retries=Config.LLM_MAX_RETRIES,
            request_timeout=120,  # 对比分析可能需要更长时间
        )

    def compare_papers(
        self,
        paper_ids: List[str],
        progress_callback=None
    ) -> str:
        """
        对比多篇论文并生成分析报告

        Args:
            paper_ids: 论文ID列表（2-5篇）
            progress_callback: 进度回调

        Returns:
            Markdown 格式的对比分析报告
        """
        num = len(paper_ids)
        logger.info(f"开始多篇对比: {num} 篇论文")

        if num < 2:
            return "错误：至少需要选择 2 篇论文进行对比"
        if num > Config.MAX_COMPARE_PAPERS:
            return f"错误：最多支持 {Config.MAX_COMPARE_PAPERS} 篇论文对比"

        # 步骤1：收集各论文的论点和信息
        paper_summaries = []
        for i, paper_id in enumerate(paper_ids):
            summary = self._get_paper_summary(paper_id)
            paper_summaries.append(summary)

            if progress_callback:
                progress_callback((i + 1) / (num + 1))

        # 步骤2：检索各论文的局限性相关内容
        for i, paper_id in enumerate(paper_ids):
            limitation_text = self._retrieve_limitations(paper_id)
            paper_summaries[i]["limitations_raw"] = limitation_text

        # 步骤3：构造 Prompt 并调用 LLM
        summaries_text = self._format_summaries(paper_summaries)
        report = self._call_llm_compare(num, summaries_text)

        if progress_callback:
            progress_callback(1.0)

        # 保存报告
        report_path = get_compare_report_path("compare")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"对比报告已保存: {report_path}")

        return report

    def _get_paper_summary(self, paper_id: str) -> Dict[str, Any]:
        """获取论文摘要信息"""
        # 从论点数据获取
        arg_path = get_argument_data_path(paper_id)
        data = load_json(arg_path) or {}

        title = data.get("paper_title", "Unknown")
        arguments = data.get("arguments", [])

        # 按类型分组论点
        claims = [a for a in arguments if a.get("type") == "claim"]
        hypotheses = [a for a in arguments if a.get("type") == "hypothesis"]
        conclusions = [a for a in arguments if a.get("type") == "conclusion"]

        return {
            "paper_id": paper_id,
            "title": title,
            "claims": claims,
            "hypotheses": hypotheses,
            "conclusions": conclusions,
            "total_arguments": len(arguments),
        }

    def _retrieve_limitations(self, paper_id: str) -> str:
        """检索论文中与局限性相关的内容"""
        queries = [
            "limitation weakness limitation drawback",
            "future work future direction",
            "assumption constraint",
        ]
        all_chunks = []
        for query in queries:
            chunks = self.indexer.search_similar(query, paper_id=paper_id, k=2)
            all_chunks.extend(chunks)

        # 去重
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            cid = chunk.metadata.get("chunk_id", "")
            if cid not in seen:
                seen.add(cid)
                unique_chunks.append(chunk)

        return "\n\n".join([c.page_content for c in unique_chunks[:6]])

    def _format_summaries(self, paper_summaries: List[Dict[str, Any]]) -> str:
        """格式化论文摘要为文本"""
        parts = []
        for i, summary in enumerate(paper_summaries):
            lines = [f"### 论文 {i+1}: {summary['title']}", ""]

            # 主张
            claims = summary.get("claims", [])
            if claims:
                lines.append("**主张 (Claims):**")
                for c in claims:
                    lines.append(f"  - {c['statement']}")
                lines.append("")

            # 假设
            hypotheses = summary.get("hypotheses", [])
            if hypotheses:
                lines.append("**假设 (Hypotheses):**")
                for h in hypotheses:
                    lines.append(f"  - {h['statement']}")
                lines.append("")

            # 结论
            conclusions = summary.get("conclusions", [])
            if conclusions:
                lines.append("**结论 (Conclusions):**")
                for c in conclusions:
                    lines.append(f"  - {c['statement']}")
                lines.append("")

            # 局限性相关内容
            lim = summary.get("limitations_raw", "")
            if lim and len(lim) > 20:
                lines.append("**局限性相关内容（原文片段）:**")
                lines.append(f"  > {lim[:500]}...")

            parts.append("\n".join(lines))

        return "\n\n---\n\n".join(parts)

    def _call_llm_compare(self, num_papers: int, summaries_text: str) -> str:
        """调用 LLM 进行对比分析"""
        # 构造表格行占位符
        paper_rows = "\n".join(
            [f"| 论文{i+1} | ... | ... | ... |" for i in range(num_papers)]
        )

        prompt = COMPARE_HUMAN.format(
            num_papers=num_papers,
            paper_summaries=summaries_text,
            paper_rows=paper_rows,
        )

        messages = [
            SystemMessage(content=COMPARE_SYSTEM),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"多篇对比 LLM 调用失败: {e}")
            return f"## 对比分析失败\n\n错误: {e}"

    def generate_research_directions(
        self, paper_ids: List[str]
    ) -> str:
        """
        仅生成研究方向建议（更聚焦的分析）

        Args:
            paper_ids: 论文ID列表

        Returns:
            Markdown 格式的方向建议
        """
        logger.info(f"生成研究方向建议: {len(paper_ids)} 篇论文")

        # 复用对比分析，但使用更聚焦的 prompt
        all_limitations = []
        for paper_id in paper_ids:
            lim_text = self._retrieve_limitations(paper_id)
            all_limitations.append(lim_text)

        combined_lim = "\n\n---\n\n".join(all_limitations)

        direction_prompt = f"""基于以下论文的局限性讨论，请提出 2-4 个具体可行的未来研究方向：

{combined_lim}

输出格式（Markdown）：

## 潜在研究方向

### 方向 1: [方向名称]
- **研究动机**: [为什么值得研究]
- **方法思路**: [大致技术路线]
- **预期贡献**: [可能的学术贡献]
- **相关论文**: [与哪篇论文的局限性相关]

### 方向 2: ...
"""

        messages = [
            SystemMessage(content="你是一个研究方向生成专家，擅长从论文局限性中发现研究机会。"),
            HumanMessage(content=direction_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"研究方向生成失败: {e}")
            return f"## 生成失败\n\n错误: {e}"
