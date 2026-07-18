"""
PaperMind 图谱构建与可视化模块
使用 NetworkX 构建论点关系图，Pyvis 生成交互式网页可视化
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from pyvis.network import Network

from src.config import RELATION_TYPE_CN, RELATION_TYPE_COLORS, RelationType
from src.utils import logger


class GraphBuilder:
    """论点关系图谱构建器"""

    # 关系类型 → 中文标签
    RELATION_LABELS = RELATION_TYPE_CN

    # 关系类型 → 颜色
    RELATION_COLORS = RELATION_TYPE_COLORS

    # 论点类型 → 节点颜色
    ARGUMENT_COLORS = {
        "claim": "#42A5F5",       # 蓝色
        "hypothesis": "#AB47BC",  # 紫色
        "conclusion": "#66BB6A",  # 绿色
    }

    # 论点类型 → 节点形状
    ARGUMENT_SHAPES = {
        "claim": "dot",
        "hypothesis": "diamond",
        "conclusion": "star",
    }

    # 论点类型 → 精简标签（显示在节点上）
    ARGUMENT_TYPE_LABELS = {
        "claim": "💡 主张",
        "hypothesis": "🔮 假设",
        "conclusion": "✅ 结论",
    }

    def build_graph(
        self,
        arguments: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        paper_title: str = "",
    ) -> nx.DiGraph:
        """
        构建论点关系有向图

        Args:
            arguments: 论点列表
            relations: 关系列表
            paper_title: 论文标题

        Returns:
            NetworkX 有向图
        """
        G = nx.DiGraph()

        # 添加节点
        for arg in arguments:
            arg_id = arg.get("id", "unknown")
            arg_type = arg.get("type", "claim")
            statement = arg.get("statement", "")
            location = arg.get("location", "Unknown")
            short_label = self.ARGUMENT_TYPE_LABELS.get(arg_type, "📌 论点")

            G.add_node(
                arg_id,
                label=short_label,
                title=self._build_node_title(arg),
                group=arg_type,
                color=self.ARGUMENT_COLORS.get(arg_type, "#78909C"),
                shape=self.ARGUMENT_SHAPES.get(arg_type, "dot"),
                arg_type=arg_type,
                full_statement=statement,
                location=location,
            )

        # 添加边
        for rel in relations:
            from_id = rel.get("from", "")
            to_id = rel.get("to", "")
            rel_type = rel.get("type", "")

            # 确保节点存在
            if from_id not in G.nodes or to_id not in G.nodes:
                continue

            G.add_edge(
                from_id,
                to_id,
                label=self.RELATION_LABELS.get(rel_type, rel_type),
                title=rel.get("explanation", ""),
                color=self.RELATION_COLORS.get(rel_type, "#999999"),
                rel_type=rel_type,
            )

        G.graph["title"] = paper_title
        logger.info(
            f"图谱构建完成: nodes={G.number_of_nodes()}, "
            f"edges={G.number_of_edges()}"
        )
        return G

    def build_multi_paper_graph(
        self,
        papers_data: List[Dict[str, Any]],
    ) -> nx.DiGraph:
        """
        构建多篇论文的联合图谱

        Args:
            papers_data: [{"title": ..., "arguments": [...], "relations": [...]}, ...]

        Returns:
            联合 NetworkX 图
        """
        G = nx.DiGraph()

        paper_colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA"]

        for pi, paper in enumerate(papers_data):
            title = paper.get("title", f"Paper {pi+1}")
            prefix = f"P{pi+1}_"
            color = paper_colors[pi % len(paper_colors)]

            # 添加论文节点
            paper_node_id = f"{prefix}paper"
            G.add_node(
                paper_node_id,
                label=self._truncate_label(title, 30),
                title=title,
                group="paper",
                color=color,
                shape="triangle",
                size=30,
                paper_title=title,
            )

            # 添加论点节点
            for arg in paper.get("arguments", []):
                arg_id = f"{prefix}{arg.get('id', 'unknown')}"
                arg_type = arg.get("type", "claim")
                short_label = self.ARGUMENT_TYPE_LABELS.get(arg_type, "📌 论点")
                G.add_node(
                    arg_id,
                    label=f"[P{pi+1}] {short_label}",
                    title=f"[{title}]<br>{arg.get('statement', '')}",
                    group=arg_type,
                    color=self.ARGUMENT_COLORS.get(arg_type, "#78909C"),
                    shape=self.ARGUMENT_SHAPES.get(arg_type, "dot"),
                )
                # 连接到论文节点
                G.add_edge(paper_node_id, arg_id, dashes=True, color="#CCC")

            # 添加关系边
            for rel in paper.get("relations", []):
                from_id = f"{prefix}{rel.get('from', '')}"
                to_id = f"{prefix}{rel.get('to', '')}"
                if from_id in G.nodes and to_id in G.nodes:
                    rel_type = rel.get("type", "")
                    G.add_edge(
                        from_id,
                        to_id,
                        label=self.RELATION_LABELS.get(rel_type, rel_type),
                        color=self.RELATION_COLORS.get(rel_type, "#999"),
                    )

        return G

    def visualize(
        self,
        G: nx.DiGraph,
        output_path: Optional[str] = None,
        height: str = "600px",
        width: str = "100%",
        bg_color: str = "#FFFFFF",
        directed: bool = True,
    ) -> str:
        """
        将 NetworkX 图可视化为交互式 HTML

        Args:
            G: NetworkX 图
            output_path: 输出 HTML 路径（可选，不存盘则返回 HTML 字符串）
            height: 画布高度
            width: 画布宽度
            bg_color: 背景色
            directed: 是否为有向图

        Returns:
            HTML 字符串
        """
        net = Network(
            height=height,
            width=width,
            bgcolor=bg_color,
            directed=directed,
        )

        # Pyvis 物理引擎配置
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.3,
              "springLength": 200,
              "springConstant": 0.04,
              "damping": 0.3
            },
            "maxVelocity": 50,
            "minVelocity": 0.75,
            "stabilization": {
              "enabled": true,
              "iterations": 200
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
          },
          "edges": {
            "smooth": {
              "type": "cubicBezier",
              "forceDirection": "vertical",
              "roundness": 0.4
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.8
              }
            }
          },
          "nodes": {
            "font": {
              "size": 14,
              "face": "Microsoft YaHei, Arial"
            }
          }
        }
        """)

        # 从 NetworkX 图添加节点和边
        net.from_nx(G)

        # add empty heading (避免 pyvis 默认标题，标题由 Streamlit 外部显示)
        net.heading = ""

        # 生成 HTML
        try:
            html = net.generate_html()

            # 注入自定义 JS：点击节点显示详情面板
            html = self._inject_click_panel(html)

            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html)
                logger.info(f"图谱已保存: {output_path}")

            return html
        except Exception as e:
            logger.error(f"图谱生成失败: {e}")
            return f"<div style='color:red'>图谱生成失败: {e}</div>"

    def visualize_to_file(
        self,
        G: nx.DiGraph,
        output_path: str,
        **kwargs,
    ):
        """保存可视化为 HTML 文件"""
        return self.visualize(G, output_path=output_path, **kwargs)

    def _truncate_label(self, text: str, max_len: int = 40) -> str:
        """截断标签文本"""
        text = text.strip()
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    def _build_node_title(self, arg: Dict[str, Any]) -> str:
        """构建节点悬浮提示文本（纯文本，vis.js 默认不渲染 HTML）"""
        lines = [
            f"ID: {arg.get('id', 'Unknown')}",
            f"类型: {arg.get('type', 'Unknown')}",
            f"论点: {arg.get('statement', '')}",
            f"位置: {arg.get('location', 'Unknown')}",
        ]
        return "\n".join(lines)

    def _inject_click_panel(self, html: str) -> str:
        """在 pyvis 生成的 HTML 中注入节点点击详情面板"""
        panel_css = """
        <style>
        #node-detail-panel {
            position: fixed;
            top: 20px;
            right: -420px;
            width: 380px;
            max-height: 70vh;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.18);
            z-index: 9999;
            padding: 20px;
            transition: right 0.35s cubic-bezier(0.4, 0, 0.2, 1);
            overflow-y: auto;
            font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
        }
        #node-detail-panel.open { right: 20px; }
        #node-detail-panel .close-btn {
            position: absolute;
            top: 8px;
            right: 12px;
            font-size: 20px;
            cursor: pointer;
            color: #999;
            border: none;
            background: none;
        }
        #node-detail-panel .close-btn:hover { color: #333; }
        #node-detail-panel .panel-type {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        #node-detail-panel .panel-statement {
            font-size: 15px;
            line-height: 1.7;
            color: #333;
            margin: 10px 0;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        #node-detail-panel .panel-meta {
            font-size: 13px;
            color: #888;
            margin-top: 12px;
        }
        #node-detail-panel .panel-placeholder {
            text-align: center;
            color: #bbb;
            font-size: 14px;
            margin-top: 40px;
        }
        </style>
        """

        panel_html = """
        <div id="node-detail-panel">
            <button class="close-btn" onclick="document.getElementById('node-detail-panel').classList.remove('open')">&times;</button>
            <div id="panel-content" class="panel-placeholder">👆 点击图中节点查看详情</div>
        </div>
        """

        click_js = """
        <script>
        (function() {
            var panel = document.getElementById('node-detail-panel');
            var content = document.getElementById('panel-content');
            var typeLabels = {claim: "💡 主张", hypothesis: "🔮 假设", conclusion: "✅ 结论"};
            var typeColors = {
                claim: {bg: "#E3F2FD", text: "#1565C0"},
                hypothesis: {bg: "#F3E5F5", text: "#7B1FA2"},
                conclusion: {bg: "#E8F5E9", text: "#2E7D32"}
            };

            // vis.js 在 pyvis 中用变量 network 存储
            if (typeof network !== 'undefined') {
                network.on('click', function(params) {
                    if (params.nodes.length > 0) {
                        var nodeId = params.nodes[0];
                        var nodeData = network.body.data.nodes.get(nodeId);
                        if (!nodeData) return;

                        var argType = nodeData.arg_type || nodeData.group || '';
                        var fullText = nodeData.full_statement || nodeData.title || '';
                        var location = nodeData.location || '';
                        var typeLabel = typeLabels[argType] || '📌 ' + argType;
                        var colors = typeColors[argType] || {bg: "#F5F5F5", text: "#666"};

                        // 清理 title 中的 HTML 标签获取纯文本
                        var plainTitle = fullText.replace(/<[^>]*>/g, '');
                        var cleanStatement = plainTitle
                            .replace(/^.*?论点:<\/b>\\s*/i, '')
                            .replace(/<b>.*?:<\\/b>/g, '')
                            .trim();

                        // 如果 full_statement 存在就用它
                        if (nodeData.full_statement) {
                            cleanStatement = nodeData.full_statement;
                        } else {
                            // 尝试从 title 中提取 (fallback)
                            var m = plainTitle.match(/论点:<\/b>\\s*(.+?)(?:<br>|$)/);
                            if (m) cleanStatement = m[1].trim();
                        }

                        content.innerHTML =
                            '<span class="panel-type" style="background:' + colors.bg + ';color:' + colors.text + '">' + typeLabel + '</span>' +
                            '<div class="panel-statement">' + cleanStatement + '</div>' +
                            (location ? '<div class="panel-meta">📍 位置: ' + location + '</div>' : '') +
                            '<div class="panel-meta">🆔 ' + nodeId + '</div>';

                        panel.classList.add('open');
                    }
                });

                // 点击空白关闭
                network.on('click', function(params) {
                    if (params.nodes.length === 0 && params.edges.length === 0) {
                        panel.classList.remove('open');
                    }
                });
            }
        })();
        </script>
        """

        # 注入到 </body> 之前
        injection = panel_css + panel_html + click_js
        html = html.replace("</body>", injection + "\n</body>")
        return html

    def get_graph_stats(self, G: nx.DiGraph) -> Dict[str, Any]:
        """获取图谱统计信息"""
        try:
            return {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": round(nx.density(G), 4),
                "is_dag": nx.is_directed_acyclic_graph(G),
                "in_degree": dict(G.in_degree()),
                "out_degree": dict(G.out_degree()),
            }
        except Exception as e:
            logger.error(f"图谱统计失败: {e}")
            return {"error": str(e)}
