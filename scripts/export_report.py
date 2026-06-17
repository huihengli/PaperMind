"""
PaperMind 报告导出脚本
将分析结果导出为 Markdown 格式
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.utils import (
    get_argument_data_path,
    get_compare_report_path,
    load_json,
    list_indexed_papers,
    logger,
)


def export_single_paper(paper_id: str, output_path: Path = None) -> str:
    """导出一篇论文的分析报告"""
    data = load_json(get_argument_data_path(paper_id))
    if not data:
        raise ValueError(f"未找到论文数据: {paper_id}")

    title = data.get("paper_title", "Unknown")
    arguments = data.get("arguments", [])
    relations = data.get("relations", [])

    lines = [
        f"# PaperMind 论文分析报告",
        f"",
        f"**论文标题**: {title}",
        f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**论点数量**: {len(arguments)}",
        f"**关系数量**: {len(relations)}",
        f"",
        f"---",
        f"",
        f"## 核心论点",
        f"",
    ]

    for i, arg in enumerate(arguments, 1):
        arg_type_emoji = {"claim": "💡", "hypothesis": "🔮", "conclusion": "✅"}
        emoji = arg_type_emoji.get(arg.get("type", ""), "📌")
        lines.append(f"### {emoji} 论点 {i}")
        lines.append(f"")
        lines.append(f"**内容**: {arg.get('statement', '')}")
        lines.append(f"**类型**: {arg.get('type', '')}")
        lines.append(f"**位置**: {arg.get('location', 'Unknown')}")
        lines.append(f"")

    if relations:
        lines.append(f"## 论点关系")
        lines.append(f"")
        for rel in relations:
            from_id = rel.get("from", "?")
            to_id = rel.get("to", "?")
            rel_type = rel.get("type", "?")

            from_arg = next((a for a in arguments if a.get("id") == from_id), None)
            to_arg = next((a for a in arguments if a.get("id") == to_id), None)

            from_text = from_arg.get("statement", from_id)[:60] if from_arg else from_id
            to_text = to_arg.get("statement", to_id)[:60] if to_arg else to_id

            lines.append(f"- **{from_text}** → [{rel_type}] → **{to_text}**")
        lines.append(f"")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"报告已保存: {output_path}")

    return report


def export_all_papers(output_dir: Path = None):
    """导出所有已索引论文的报告"""
    if output_dir is None:
        output_dir = Config.REPORTS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    papers = list_indexed_papers()
    logger.info(f"导出 {len(papers)} 篇论文的报告")

    for paper in papers:
        pid = paper["id"]
        try:
            report = export_single_paper(pid)
            output_path = output_dir / f"report_{pid}.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"✅ {paper.get('title', 'Unknown')[:50]} -> {output_path}")
        except Exception as e:
            print(f"❌ {paper.get('title', 'Unknown')[:50]}: {e}")


if __name__ == "__main__":
    export_all_papers()
