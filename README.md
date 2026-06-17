# 📄 PaperMind — 科研论文智能分析助手

> **不只是检索论文，而是理解科研脉络**

PaperMind 是一个基于 RAG（检索增强生成）架构的科研论文智能分析助手。它能自动从 PDF 论文中抽取核心论点、识别论点间的逻辑关系、支持多篇论文对比分析，并生成未来研究方向的建议。

---

## ✨ 核心功能

| 功能 | 描述 |
|------|------|
| 📤 **PDF 上传与解析** | 支持单篇/批量上传，自动提取文本、章节和元数据 |
| 🔍 **智能索引** | 向量化存储，毫秒级语义检索，支持按论文字段过滤 |
| 💡 **论点抽取** | LLM 自动提取 3-7 个核心论点，分类为主张/假设/结论 |
| 🔗 **关系识别** | 识别论点间的支持、矛盾、细化、因果关系 |
| 📊 **可视化** | 交互式论点网络图，可拖拽、点击查看详情 |
| 🔄 **多篇对比** | 2-5 篇论文的论点对比、局限性归纳、差异分析 |
| 💡 **方向生成** | 基于研究空白自动生成可行研究方向建议 |
| 💬 **智能问答** | 基于 RAG 的自然语言问答，答案可追溯到原文 |
| 📥 **报告导出** | 一键导出分析报告为 Markdown |

---

## 🚀 快速开始

### 1. 克隆项目

```bash
cd papermind
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 OpenAI API Key
```

`.env` 配置示例：

```env
OPENAI_API_KEY=sk-your-api-key-here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

---

## 📁 项目结构

```
PaperMind/
├── app.py                         # Streamlit 主入口
├── requirements.txt               # Python 依赖
├── .env.example                   # 环境变量模板
├── README.md                      # 项目说明
│
├── src/                           # 核心源码
│   ├── config.py                  # 配置管理
│   ├── pdf_loader.py              # PDF 解析
│   ├── indexer.py                 # 向量索引
│   ├── argument_extractor.py      # 论点抽取
│   ├── relation_identifier.py     # 关系识别
│   ├── rag_chain.py               # RAG 问答
│   ├── multi_paper_analyzer.py    # 多篇对比
│   ├── graph_builder.py           # 图谱可视化
│   └── utils.py                   # 工具函数
│
├── scripts/                       # 辅助脚本
│   ├── batch_index.py             # 批量索引
│   └── export_report.py           # 报告导出
│
├── tests/                         # 测试
│   ├── test_pdf_loader.py
│   └── test_argument_extractor.py
│
├── data/                          # 数据目录
│   ├── raw_pdfs/                  # 原始 PDF
│   ├── parsed_json/               # 分析结果
│   └── reports/                   # 导出报告
│
└── chroma_db/                     # 向量数据库（自动生成）
```

---

## 🛠 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 前端 | Streamlit | 快速构建交互式 Web 界面 |
| LLM 框架 | LangChain | RAG 流程编排 |
| 向量数据库 | Chroma | 文档块存储与语义检索 |
| Embedding | OpenAI text-embedding-3-small | 文本向量化 |
| LLM | GPT-4o-mini / GPT-3.5-turbo | 论点抽取、关系识别、对比分析 |
| PDF 解析 | pdfplumber + PyPDF2 | 论文文本提取 |
| 图谱 | NetworkX + Pyvis | 论点关系建模与交互式可视化 |

---

## 📖 使用指南

### 单篇论文分析

1. 上传 PDF 论文
2. 等待自动索引和分析（约 30-60 秒）
3. 点击左侧论文名查看：
   - **论点网络图**：可拖拽的交互式关系图
   - **论点详情**：每个论点的完整信息和关联关系
   - **统计信息**：论点和关系分布

### 多篇论文对比

1. 在左侧勾选 2-5 篇论文
2. 进入「多篇对比」页面
3. 点击「开始对比分析」
4. 获取包含论点对比、共同点、差异点、研究空白的完整报告

### 智能问答

1. 进入「问答交互」页面
2. 选择论文范围（全部或特定论文）
3. 输入问题或点击快捷问题
4. 回答会附带原文来源引用

---

## 🧪 运行测试

```bash
pytest tests/ -v
```

---

## ⚠️ 注意事项

1. **API Key**：需要有效的 OpenAI API Key，请确保账户有余额
2. **PDF 格式**：优先使用文字版 PDF（非扫描版），扫描版解析效果有限
3. **成本控制**：每篇论文分析约消耗 2K-5K token，建议使用 `gpt-4o-mini` 控制成本
4. **论文质量**：分析质量取决于原文质量，建议优先上传结构化良好的学术论文

---

## 🔮 未来规划

- [ ] OCR 支持扫描版 PDF
- [ ] 支持更多 LLM（Claude、本地模型）
- [ ] 论文推荐系统
- [ ] 文献综述自动生成
- [ ] 引用网络分析

---

## 📄 License

MIT License
