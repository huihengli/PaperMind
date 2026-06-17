"""
PaperMind - 科研论文智能分析助手
Streamlit 主入口应用

功能模块：
- 论文上传与管理
- 单篇论点分析 + 关系图谱可视化
- 多篇论文对比分析
- RAG 问答交互
- 报告导出
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from streamlit.components.v1 import html as st_html

# 必须在导入其他模块前设置页面配置
st.set_page_config(
    page_title="PaperMind - 科研论文智能分析助手",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 本地模块导入
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config, PaperStatus, RelationType
from src.pdf_loader import PDFLoader
from src.indexer import PaperIndexer
from src.argument_extractor import ArgumentExtractor
from src.relation_identifier import RelationIdentifier
from src.rag_chain import RAGChain
from src.multi_paper_analyzer import MultiPaperAnalyzer
from src.graph_builder import GraphBuilder
from src.utils import (
    list_indexed_papers,
    delete_paper_data,
    load_json,
    get_argument_data_path,
    logger,
)


# ============================================================
# CSS 样式
# ============================================================

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .paper-card {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 8px;
        transition: all 0.2s;
    }
    .paper-card:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-completed { background: #E8F5E9; color: #2E7D32; }
    .status-indexing { background: #FFF3E0; color: #E65100; }
    .status-failed { background: #FFEBEE; color: #C62828; }
    .status-uploaded { background: #E3F2FD; color: #1565C0; }
    .arg-claim { color: #1565C0; font-weight: 500; }
    .arg-hypothesis { color: #7B1FA2; font-weight: 500; }
    .arg-conclusion { color: #2E7D32; font-weight: 500; }
    .source-box {
        background: #F5F5F5;
        border-left: 3px solid #667eea;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
"""


# ============================================================
# 初始化服务（缓存）
# ============================================================

@st.cache_resource
def init_services():
    """初始化所有核心服务实例"""
    try:
        Config.validate()
    except ValueError as e:
        st.error(f"配置错误: {e}")
        st.info("请在项目根目录创建 .env 文件并设置 OPENAI_API_KEY")
        st.stop()

    pdf_loader = PDFLoader()
    indexer = PaperIndexer()
    arg_extractor = ArgumentExtractor(indexer)
    rel_identifier = RelationIdentifier()
    rag_chain = RAGChain(indexer)
    multi_analyzer = MultiPaperAnalyzer(indexer, arg_extractor)
    graph_builder = GraphBuilder()

    return {
        "pdf_loader": pdf_loader,
        "indexer": indexer,
        "arg_extractor": arg_extractor,
        "rel_identifier": rel_identifier,
        "rag_chain": rag_chain,
        "multi_analyzer": multi_analyzer,
        "graph_builder": graph_builder,
    }


# ============================================================
# 辅助函数
# ============================================================

def process_pdf(uploaded_file, services, progress_bar, status_text):
    """处理上传的 PDF：解析 + 索引 + 分析"""
    filename = uploaded_file.name

    # 重复检测：检查是否有同名已完成论文
    existing_papers = list_indexed_papers()
    for p in existing_papers:
        if p.get("filename") == filename:
            st.warning(f"⚠️ 论文「{filename}」已存在，跳过重复上传")
            return None

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        # 步骤1: 解析 PDF
        status_text.text("正在解析 PDF...")
        progress_bar.progress(20)
        paper = services["pdf_loader"].load_pdf(tmp_path)
        paper_id = paper["id"]

        # 步骤2: 获取全文
        status_text.text("正在提取全文...")
        progress_bar.progress(40)
        full_text = services["pdf_loader"].get_full_text(paper_id)

        if not full_text:
            raise ValueError("无法提取论文文本")

        # 步骤3: 构建索引
        status_text.text("正在构建向量索引...")
        progress_bar.progress(60)
        services["indexer"].index_paper(paper_id, full_text)

        # 步骤4: 抽取论点
        status_text.text("正在抽取核心论点...")
        progress_bar.progress(75)
        arguments = services["arg_extractor"].extract_arguments(
            paper_id, paper.get("title", "Unknown")
        )

        # 步骤5: 识别关系
        if arguments:
            status_text.text("正在识别论点关系...")
            progress_bar.progress(90)
            services["rel_identifier"].identify_relations(
                paper_id, paper.get("title", "Unknown"), arguments
            )

        progress_bar.progress(100)
        status_text.text("处理完成！")
        time.sleep(0.5)

        return paper_id

    finally:
        # 清理临时文件
        if tmp_path.exists():
            tmp_path.unlink()


# ============================================================
# UI 组件
# ============================================================

def render_sidebar(services):
    """渲染侧边栏 - 论文库和导航"""
    st.sidebar.markdown("## 📁 论文库")

    papers = list_indexed_papers()

    # 上传区域
    with st.sidebar.expander("📤 上传论文", expanded=len(papers) == 0):
        uploaded_files = st.file_uploader(
            "拖拽或点击上传 PDF",
            type=["pdf"],
            accept_multiple_files=True,
            help="支持单篇或批量上传",
            label_visibility="collapsed",
        )

        if uploaded_files:
            if st.button("🚀 开始处理", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                has_new = False

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"处理中 ({i+1}/{len(uploaded_files)}): {uploaded_file.name}")
                    try:
                        result = process_pdf(uploaded_file, services, progress_bar, status_text)
                        if result is not None:
                            has_new = True
                    except Exception as e:
                        st.error(f"处理失败 [{uploaded_file.name}]: {e}")
                        logger.error(f"PDF 处理失败: {e}")

                progress_bar.empty()
                status_text.empty()
                if has_new:
                    st.rerun()

    # 论文列表
    st.sidebar.markdown("---")
    if not papers:
        st.sidebar.info("📭 暂无论文，请上传 PDF 开始使用")
        return None, None

    st.sidebar.markdown(f"**共 {len(papers)} 篇论文**")

    selected_paper_id = st.session_state.get("selected_paper_id", None)
    selected_papers_for_compare = st.session_state.get("selected_papers_for_compare", [])

    for paper in papers:
        pid = paper["id"]
        status = paper.get("status", "uploaded")
        status_class = f"status-{status}"

        col1, col2, col3 = st.sidebar.columns([0.68, 0.17, 0.15])

        with col1:
            if st.button(
                f"📄 {paper.get('title', 'Unknown')[:40]}",
                key=f"paper_{pid}",
                use_container_width=True,
                help=f"状态: {status} | 页数: {paper.get('page_count', '?')}",
            ):
                st.session_state["selected_paper_id"] = pid
                selected_paper_id = pid
                st.rerun()

        with col2:
            is_checked = pid in selected_papers_for_compare
            if st.checkbox(" ", value=is_checked, key=f"cmp_{pid}", label_visibility="collapsed"):
                if pid not in selected_papers_for_compare:
                    selected_papers_for_compare.append(pid)
            else:
                if pid in selected_papers_for_compare:
                    selected_papers_for_compare.remove(pid)

        with col3:
            if st.button("🗑️", key=f"del_{pid}", help=f"删除「{paper.get('title', '')[:30]}」"):
                # 从向量库中删除
                services["indexer"].delete_paper(pid)
                # 删除本地数据文件
                delete_paper_data(pid)
                # 清理 session state
                if st.session_state.get("selected_paper_id") == pid:
                    st.session_state["selected_paper_id"] = None
                if pid in selected_papers_for_compare:
                    selected_papers_for_compare.remove(pid)
                st.sidebar.success(f"已删除: {paper.get('title', 'Unknown')[:30]}")
                time.sleep(0.5)
                st.rerun()

    st.session_state["selected_papers_for_compare"] = selected_papers_for_compare

    # 底部导航
    st.sidebar.markdown("---")
    nav = st.sidebar.radio(
        "📊 功能导航",
        ["📖 单篇分析", "🔄 多篇对比", "💬 问答交互"],
        key="nav_mode",
    )

    return selected_paper_id, nav


def render_single_analysis(paper_id, services):
    """渲染单篇论文分析页面"""
    if not paper_id:
        st.info("👈 请从左侧论文库选择一篇论文")
        return

    # 加载数据
    arg_path = get_argument_data_path(paper_id)
    data = load_json(arg_path)

    if not data:
        st.warning("该论文尚未完成分析，请重新上传处理")
        return

    paper_title = data.get("paper_title", "Unknown")
    arguments = data.get("arguments", [])
    relations = data.get("relations", [])

    st.markdown(f"### 📖 {paper_title}")

    # 标签页
    tab1, tab2, tab3 = st.tabs(["🔗 论点网络图", "📋 论点详情", "📊 统计信息"])

    with tab1:
        if not arguments:
            st.info("未抽取到论点")
        else:
            # 构建图谱
            G = services["graph_builder"].build_graph(arguments, relations, paper_title)
            graph_html = services["graph_builder"].visualize(
                G, height="550px", bg_color="#FAFAFA"
            )

            # 保存到临时文件并用 iframe 渲染
            tmp_path = Path(tempfile.gettempdir()) / f"papermind_graph_{paper_id}.html"
            services["graph_builder"].visualize_to_file(G, str(tmp_path), height="550px")

            with open(tmp_path, "r", encoding="utf-8") as f:
                graph_html = f.read()

            st_html(graph_html, height=580, scrolling=True)

            # 图例
            st.markdown("**图例**")
            cols = st.columns(7)
            with cols[0]: st.markdown("🔵 主张")
            with cols[1]: st.markdown("🟣 假设")
            with cols[2]: st.markdown("🟢 结论")
            with cols[3]: st.markdown("🟢→ 支持")
            with cols[4]: st.markdown("🔴→ 矛盾")
            with cols[5]: st.markdown("🔵→ 细化")
            with cols[6]: st.markdown("🟠→ 因果")

    with tab2:
        st.markdown("#### 📋 论点列表")
        for arg in arguments:
            arg_type = arg.get("type", "claim")
            type_class = f"arg-{arg_type}"
            type_emoji = {"claim": "💡", "hypothesis": "🔮", "conclusion": "✅"}.get(arg_type, "📌")

            with st.expander(f"{type_emoji} {arg.get('statement', '')[:80]}..."):
                st.markdown(f"**完整论点**: {arg.get('statement', '')}")
                st.markdown(f"**类型**: <span class='{type_class}'>{arg_type}</span>", unsafe_allow_html=True)
                st.markdown(f"**位置**: {arg.get('location', 'Unknown')}")

                # 显示相关关系
                arg_id = arg.get("id")
                related_relations = [
                    r for r in relations
                    if r.get("from") == arg_id or r.get("to") == arg_id
                ]
                if related_relations:
                    st.markdown("**关联关系**:")
                    for rel in related_relations:
                        from_id = rel.get("from")
                        to_id = rel.get("to")
                        rel_type = rel.get("type", "")
                        direction = "→" if from_id == arg_id else "←"
                        other_id = to_id if from_id == arg_id else from_id

                        # 找到对方论点
                        other_arg = next(
                            (a for a in arguments if a.get("id") == other_id), None
                        )
                        other_text = other_arg.get("statement", other_id)[:60] if other_arg else other_id

                        st.markdown(
                            f"  {direction} [{rel_type}] {other_text}..."
                        )

    with tab3:
        stats = {
            "论点数": len(arguments),
            "关系数": len(relations),
            "主张 (Claim)": sum(1 for a in arguments if a.get("type") == "claim"),
            "假设 (Hypothesis)": sum(1 for a in arguments if a.get("type") == "hypothesis"),
            "结论 (Conclusion)": sum(1 for a in arguments if a.get("type") == "conclusion"),
        }

        col1, col2, col3 = st.columns(3)
        metrics = list(stats.items())
        for i, (key, val) in enumerate(metrics[:3]):
            with [col1, col2, col3][i]:
                st.metric(key, val)

        col1, col2 = st.columns(2)
        for i, (key, val) in enumerate(metrics[3:]):
            with [col1, col2][i]:
                st.metric(key, val)

        # 关系类型分布
        if relations:
            st.markdown("#### 关系类型分布")
            from collections import Counter
            rel_counts = Counter(r.get("type", "unknown") for r in relations)
            st.bar_chart(rel_counts)


def render_multi_compare(services):
    """渲染多篇对比页面"""
    st.markdown("### 🔄 多篇论文对比分析")

    selected_ids = st.session_state.get("selected_papers_for_compare", [])

    if len(selected_ids) < 2:
        st.info("👈 请在左侧论文库勾选至少 2 篇论文（点击论文旁的复选框）")
        return

    if len(selected_ids) > Config.MAX_COMPARE_PAPERS:
        st.warning(f"最多支持 {Config.MAX_COMPARE_PAPERS} 篇论文对比，请取消勾选部分论文")
        return

    # 显示已选择的论文
    papers = list_indexed_papers()
    selected_papers = [p for p in papers if p["id"] in selected_ids]

    st.markdown("**已选择的论文:**")
    for p in selected_papers:
        st.markdown(f"- 📄 {p.get('title', 'Unknown')[:60]}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 开始对比分析", use_container_width=True, type="primary"):
            with st.spinner("正在分析中，这可能需要 1-2 分钟..."):
                try:
                    progress_bar = st.progress(0)

                    def update_progress(p):
                        progress_bar.progress(int(p * 100))

                    report = services["multi_analyzer"].compare_papers(
                        selected_ids, progress_callback=update_progress
                    )
                    st.session_state["compare_report"] = report
                    progress_bar.progress(100)
                except Exception as e:
                    st.error(f"对比分析失败: {e}")
                    logger.error(f"对比分析失败: {e}")

    with col2:
        if st.button("💡 仅生成研究方向", use_container_width=True):
            with st.spinner("正在分析研究空白..."):
                try:
                    directions = services["multi_analyzer"].generate_research_directions(
                        selected_ids
                    )
                    st.session_state["compare_report"] = directions
                except Exception as e:
                    st.error(f"方向生成失败: {e}")

    # 显示报告
    report = st.session_state.get("compare_report", "")
    if report:
        st.markdown("---")
        st.markdown(report)

        # 导出按钮
        st.download_button(
            "📥 导出报告 (Markdown)",
            data=report,
            file_name=f"PaperMind_对比报告_{time.strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )


def render_qa_interface(paper_id, services):
    """渲染问答交互页面"""
    st.markdown("### 💬 论文问答")

    # 论文选择
    papers = list_indexed_papers()
    paper_options = {"全部论文": None}
    for p in papers:
        paper_options[p.get("title", "Unknown")[:60]] = p["id"]

    selected_title = st.selectbox(
        "选择要提问的论文范围",
        list(paper_options.keys()),
        index=0,
    )
    selected_pid = paper_options[selected_title]

    # 快捷问题
    st.markdown("**快速提问:**")
    quick_questions = [
        "这篇论文的核心创新是什么？",
        "论文提出了什么方法？效果如何？",
        "作者提到了哪些局限性？",
        "这篇论文和其他相关工作有什么不同？",
        "论文的实验设计是什么？用了什么数据集？",
    ]

    cols = st.columns(len(quick_questions))
    q_input = st.text_input("输入你的问题", placeholder="例如：这篇论文的核心贡献是什么？")

    for i, q in enumerate(quick_questions):
        with cols[i]:
            if st.button(q[:20] + "...", key=f"qq_{i}", use_container_width=True):
                q_input = q
                st.session_state["current_question"] = q

    # 问答历史
    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    # 提交问题
    if st.button("🔍 提问", type="primary") or st.session_state.get("current_question"):
        question = q_input or st.session_state.get("current_question", "")
        if question:
            with st.spinner("思考中..."):
                result = services["rag_chain"].query(
                    question,
                    paper_id=selected_pid,
                )
                st.session_state["qa_history"].append(result)
                st.session_state.pop("current_question", None)

    # 显示历史回答
    for i, item in enumerate(reversed(st.session_state["qa_history"])):
        with st.chat_message("user"):
            st.markdown(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            # 显示来源
            if item.get("sources"):
                with st.expander("📎 查看原文来源"):
                    for j, src in enumerate(item["sources"]):
                        st.markdown(
                            f'<div class="source-box">'
                            f'<b>来源 {j+1}:</b> {src["paper_title"]} | '
                            f'{src["section"]} | Page {src["page"]}<br>'
                            f'{src["text"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # 清除历史
    if st.session_state["qa_history"]:
        if st.button("🗑️ 清除对话历史"):
            st.session_state["qa_history"] = []
            services["rag_chain"].clear_history()
            st.rerun()


# ============================================================
# 主函数
# ============================================================

def main():
    """PaperMind 主入口"""
    # 自定义 CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 页头
    st.markdown('<div class="main-header">📄 PaperMind</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">不只是检索论文，而是理解科研脉络</div>',
        unsafe_allow_html=True,
    )

    # 初始化服务
    with st.spinner("正在初始化服务..."):
        try:
            services = init_services()
        except Exception as e:
            st.error(f"服务初始化失败: {e}")
            st.stop()

    # 初始化 session state
    if "selected_paper_id" not in st.session_state:
        st.session_state["selected_paper_id"] = None
    if "selected_papers_for_compare" not in st.session_state:
        st.session_state["selected_papers_for_compare"] = []

    # 渲染侧边栏
    selected_paper_id, nav = render_sidebar(services)

    # 确保 paper_id 同步
    if selected_paper_id:
        st.session_state["selected_paper_id"] = selected_paper_id
    else:
        selected_paper_id = st.session_state.get("selected_paper_id")

    # 渲染主内容区
    if nav == "📖 单篇分析":
        render_single_analysis(selected_paper_id, services)
    elif nav == "🔄 多篇对比":
        render_multi_compare(services)
    elif nav == "💬 问答交互":
        render_qa_interface(selected_paper_id, services)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    main()
