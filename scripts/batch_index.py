"""
PaperMind 批量索引脚本
用于批量索引已有的 PDF 论文文件
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.pdf_loader import PDFLoader
from src.indexer import PaperIndexer
from src.argument_extractor import ArgumentExtractor
from src.relation_identifier import RelationIdentifier
from src.utils import logger as log_utils

logger = log_utils


def main():
    parser = argparse.ArgumentParser(description="PaperMind 批量索引脚本")
    parser.add_argument(
        "pdf_dir",
        type=str,
        help="包含 PDF 文件的目录路径",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="跳过论点抽取和关系识别",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录中的 PDF",
    )

    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"目录不存在: {pdf_dir}")
        sys.exit(1)

    # 收集 PDF 文件
    if args.recursive:
        pdf_files = list(pdf_dir.rglob("*.pdf"))
    else:
        pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"目录中未找到 PDF 文件: {pdf_dir}")
        sys.exit(1)

    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")

    # 初始化服务
    Config.validate()
    loader = PDFLoader()
    indexer = PaperIndexer()
    arg_extractor = ArgumentExtractor(indexer)
    rel_identifier = RelationIdentifier()

    # 处理每个 PDF
    success_count = 0
    for i, pdf_path in enumerate(pdf_files):
        logger.info(f"处理 ({i+1}/{len(pdf_files)}): {pdf_path.name}")
        try:
            # 解析 PDF
            paper = loader.load_pdf(pdf_path)
            paper_id = paper["id"]

            # 获取全文
            full_text = loader.get_full_text(paper_id)
            if not full_text:
                logger.error(f"无法提取文本: {pdf_path.name}")
                continue

            # 索引
            indexer.index_paper(paper_id, full_text)

            # 分析
            if not args.skip_analysis:
                arguments = arg_extractor.extract_arguments(
                    paper_id, paper.get("title", "Unknown")
                )
                if arguments:
                    rel_identifier.identify_relations(
                        paper_id, paper.get("title", "Unknown"), arguments
                    )

            success_count += 1
            logger.info(f"✅ 处理成功: {paper.get('title', pdf_path.stem)}")

        except Exception as e:
            logger.error(f"❌ 处理失败 [{pdf_path.name}]: {e}")

    logger.info(f"批量索引完成: {success_count}/{len(pdf_files)} 成功")


if __name__ == "__main__":
    main()
