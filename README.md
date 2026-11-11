# 📚 企业知识库问答系统（RAG）

基于 LangChain + Chroma + DeepSeek 的企业知识库问答系统，支持混合检索、Cross-Encoder 重排、多轮对话与完整评估实验。

---

## 📖 目录

- [项目简介](#-项目简介)
- [功能特性](#-功能特性)
- [项目结构](#-项目结构)
- [环境要求](#-环境要求)
- [安装步骤](#-安装步骤)
- [配置说明](#-配置说明)
- [快速开始](#-快速开始)
- [核心模块](#-核心模块)
- [评估实验](#-评估实验)
- [常见问题](#-常见问题)
- [技术栈](#-技术栈)
- [License](#-license)

---

## 🎯 项目简介

本项目面向小微企业知识库场景，将企业内部的规章制度、产品说明、FAQ 等文档，构建成一个可对话的智能问答系统。

**核心流程**：

```
文档加载 → 切分 → 向量化 → 混合检索 → Cross-Encoder 重排 → LLM 生成
```

**适用场景**：

- 企业员工自助问答（规章制度、报销、IT 支持）
- 产品文档智能检索
- 客服知识库自动回复

---

## ✨ 功能特性

| 功能 | 说明 |
|:---|:---|
| 多格式文档支持 | `.md` 和 `.pdf` |
| Markdown 感知切分 | MD 文件先按标题切，再按字符数细切 |
| 混合检索 | 向量检索 + BM25 关键词检索，RRF 融合 |
| Cross-Encoder 重排 | `BAAI/bge-reranker-v2-m3` 精排 top-3 |
| 多轮对话 | 保留最近 3 轮上下文，支持指代理解 |
| 流式输出 | DeepSeek 返回文本片段时逐步显示回答 |
| 对话历史导出 | 下载当前会话的 Markdown 问答记录，保留回答中的引用 |
| 多用户会话隔离 | 每个浏览器会话分别维护对话记录和检索设置 |
| 来源可追溯 | 回答末尾标注来源，前端可展开查看原文 |
| 参数可调 | 召回策略、top_n、召回候选数 k 前端可调 |
| 完整评估 | 四组对比实验 + 可视化报告 |

---

## 📂 项目结构

```
rag_company_FAQ/
├── app.py                      # Streamlit 前端
├── rag_engine.py               # RAG 核心引擎
├── evaluate.py                 # 评估脚本
├── generate_report.py          # 评估报告生成器
├── build_all_indexes.py        # 批量建库脚本
├── eval_set.json               # 评估集（30 条问答对）
├── evaluation_result.json      # 评估结果
├── requirements.txt            # 依赖清单
├── .env                        # 环境变量（不提交）
├── .gitignore                  # Git 忽略规则
├── README.md                   # 项目说明
├── corpus/                     # 文档目录
│   ├── 员工手册_v2.3.md
│   ├── 代码提交规范.md
│   ├── 分支管理规范.md
│   ├── API设计规范.md
│   ├── 数据库设计规范.md
│   ├── 需求评审流程.md
│   ├── 开发与测试流程.md
│   ├── 上线发布流程.md
│   ├── 故障应急流程.md
│   ├── 拾级CRM产品说明.md
│   ├── 拾级ERP产品说明.md
│   ├── 拾级AI平台说明.md
│   ├── 产品定价方案.md
│   ├── 新员工FAQ.md
│   ├── IT支持FAQ.md
│   └── 报销FAQ.md
├── chroma_db_v4/               # v4 向量库（不提交）
├── chroma_db_bge_m3/           # bge-m3 向量库（不提交）
├── report_YYYYMMDD_HHMMSS/     # 自动生成的报告目录
│   ├── evaluation_report_*.md
│   └── charts/
└── logs/                       # 日志目录
```

---

## 🖥️ 环境要求

| 项目 | 要求 |
|:---|:---|
| Python | 3.10+ |
| 内存 | ≥ 8GB（使用 bge-m3 时建议 16GB） |
| 磁盘 | ≥ 5GB（含模型缓存） |
| 操作系统 | Windows / macOS / Linux |

---

## 🚀 安装步骤

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd rag_company_FAQ
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 准备文档

把企业文档放入 `corpus/` 目录，支持 `.md` 和 `.pdf`。

---

## ⚙️ 配置说明

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key
```

| 变量 | 用途 | 获取地址 |
|:---|:---|:---|
| `DEEPSEEK_API_KEY` | DeepSeek 大模型调用 | https://platform.deepseek.com |
| `DASHSCOPE_API_KEY` | 百炼 `text-embedding-v4` 向量化 | https://bailian.console.aliyun.com |

### `requirements.txt`

```txt
streamlit>=1.30
langchain>=0.2
langchain-community>=0.2
langchain-chroma>=0.1
langchain-deepseek>=0.1
langchain-text-splitters>=0.2
chromadb>=0.4
dashscope>=1.14
python-dotenv>=1.0
pymupdf>=1.24
sentence-transformers>=2.6
matplotlib>=3.8
pandas>=2.0
openpyxl>=3.1
```

---

## 🏃 快速开始

### 1. 建库

```bash
python build_all_indexes.py
```

会分别为两种向量模型建立向量库：

| 模型 | 向量库目录 |
|:---|:---|
| `text-embedding-v4` | `chroma_db_v4/` |
| `bge-m3` | `chroma_db_bge_m3/` |

> 首次建库较慢，需要向量化所有文档。之后可复用。

### 2. 启动问答系统

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

### 3. 前端使用

侧边栏可调整：

| 控件 | 作用 | 范围 |
|:---|:---|:---|
| 召回策略 | 纯向量 / 纯 BM25 / 混合 / 混合+重排 | 四选一 |
| top_n | 最终送给 LLM 的文档数 | 1–10 |
| 召回候选数 k | 混合检索时每路召回候选数 | 5–30 |

主区域支持：

- 多轮对话（保留最近 3 轮上下文）
- 流式显示回答，生成完成后可展开查看来源
- 每个回答下方展开查看检索到的原文片段和章节路径
- 一键清空对话

侧边栏的 **导出对话（Markdown）** 可下载当前会话已完成的问答，文件采用 UTF-8 编码。清空对话后，导出按钮恢复为禁用状态。

会话以 Streamlit 浏览器连接为单位。分别打开两个标签页即可独立提问、调整检索参数、导出和清空各自的记录。刷新页面会建立新会话，历史保存在当前连接的内存中。各会话共用同一个企业知识库和模型。

---

## 🧩 核心模块

### `rag_engine.py`

RAG 核心引擎，封装完整流程。

| 方法 | 职责 |
|:---|:---|
| `load_documents()` | 加载 `corpus/` 下的 `.md` 和 `.pdf` |
| `split_documents()` | MD 按标题切 + 字符数细切；PDF 按字符数切 |
| `build_index()` | 从零建库，清空旧库后向量化存入 Chroma |
| `load_existing_index()` | 加载已有向量库，重建 BM25 和检索链 |
| `retrieve()` | 执行完整检索链（混合召回 → 重排） |
| `generate()` | 基于上下文生成回答 |
| `query()` | 完整问答流程 |
| `debug_retrieve()` | 调试：查看混合召回和重排结果 |

**关键设计**：

- 向量检索召回 20 个候选
- BM25 召回 20 个候选
- RRF 融合后交给 Cross-Encoder 重排
- 重排后取 top_n 送给 LLM

### `app.py`

Streamlit 前端，提供交互界面。

- 侧边栏：召回策略、top_n、召回候选数 k、清空对话、导出对话
- 主区域：多轮对话、来源展开
- 上下文记忆：保留最近 3 轮

### `evaluate.py`

四组对比实验，输出 `evaluation_result.json`。

### `generate_report.py`

读取 `evaluation_result.json`，生成：

- 带时间戳的 Markdown 报告
- 可视化图表（分组柱状图 / 横向柱状图）
- MRR 单独出图，纵轴固定 0–1

---

## 📊 评估实验

### 实验设计

| 实验 | 对比对象 |
|:---|:---|
| 实验 1 | 纯向量 / 纯 BM25 / 混合 / 混合+重排 |
| 实验 2 | 五种切分方法（200/0、500/50、1000/100、按句号、按段落） |
| 实验 3 | `text-embedding-v4` vs `bge-m3` |
| 实验 4 | 无约束 Prompt vs 带引用+拒答约束 Prompt |

### 评估指标

| 指标 | 含义 |
|:---|:---|
| Recall@3 | 前 3 个检索结果中包含正确答案的比例 |
| MRR | 正确答案排名的倒数平均值 |
| 引用准确率 | 引用来源与标准来源匹配的比例 |
| 答案正确率 | 回答包含正确答案的比例 |
| 漏答率 | 有答案类问题中被拒答的比例 |
| 有答案类幻觉率 | 有答案类问题中编造答案的比例 |
| 无答案类幻觉率 | 无答案类问题中硬编答案的比例 |
| 总幻觉率 | 所有编造答案数 / 评估集总数 |
| 拒答准确率 | 无答案类问题中正确拒答的比例 |

### 评估结果摘要

**实验 1：召回策略**

| 策略 | Recall@3 | MRR | 总幻觉率 | 平均耗时 |
|:---|:---:|:---:|:---:|:---:|
| 纯向量检索 | 100.00% | 0.9386 | 0.00% | 0.135s |
| 纯 BM25 检索 | 5.26% | 0.0526 | 0.00% | 0.000s |
| 混合检索 | 94.74% | 0.8596 | 0.00% | 0.131s |
| 混合检索 + 重排 | 100.00% | **0.9737** | 0.00% | 2.526s |

**实验 2：切分方法**

| 策略 | Recall@3 | MRR | 总幻觉率 | 平均耗时 |
|:---|:---:|:---:|:---:|:---:|
| chunk_size=200, overlap=0 | 100.00% | 1.0000 | 0.00% | 3.187s |
| chunk_size=500, overlap=50 | 100.00% | 1.0000 | 3.33% | 6.707s |
| chunk_size=1000, overlap=100 | 100.00% | 1.0000 | 3.33% | 11.178s |
| 按句号切（200） | 100.00% | 1.0000 | 3.33% | 9.897s |
| 按段落切（1000） | 100.00% | 1.0000 | 0.00% | 11.207s |

**实验 3：向量化模型**

| 模型 | Recall@3 | MRR | 答案正确率 | 成本 |
|:---|:---:|:---:|:---:|:---:|
| text-embedding-v4 | 100.00% | 0.9737 | 89.47% | $0.0006 |
| bge-m3 | 100.00% | 0.9737 | **94.74%** | **免费** |

**实验 4：防幻觉 Prompt**

| 策略 | 总幻觉率 | 无答案类幻觉率 | 拒答准确率 | 答案正确率 |
|:---|:---:|:---:|:---:|:---:|
| 无约束 Prompt | 26.67% | 72.73% | 27.27% | 100.00% |
| RAG + 引用 + 拒答约束 | **0.00%** | **0.00%** | **100.00%** | 84.21% |

### 主要结论

1. **召回策略**：混合检索 + 重排 MRR 最高（0.9737），纯向量检索性价比最高，纯 BM25 几乎失效。
2. **切分方法**：chunk_size=200 幻觉率最低且耗时较短；chunk_size 越大，建库耗时越高。
3. **向量模型**：bge-m3 免费且答案正确率更高（94.74% vs 89.47%），推荐优先使用。
4. **防幻觉**：带引用与拒答约束的 Prompt 能把总幻觉率从 26.67% 降到 0%，生产环境必须启用。

---

## ❓ 常见问题

### Q1：向量库和 Embedding 模型不匹配怎么办？

向量库是用某个模型建的，查询时必须用同一个模型。项目里向量库目录和模型是绑定的：

| 模型 | 向量库目录 |
|:---|:---|
| `text-embedding-v4` | `chroma_db_v4` |
| `bge-m3` | `chroma_db_bge_m3` |

### Q2：第一次运行很慢？

- 首次加载 `bge-m3` 会下载约 2GB 模型；
- 首次建库需要向量化所有文档；
- 之后会走缓存，速度正常。

### Q3：`纯 BM25 检索` 策略下 `k` 参数不生效？

因为 BM25 直接截取 `top_n`，不走混合召回的 `k`。

### Q4：如何切换向量库？

修改 `app.py` 中 `load_engine()` 的 `persist_dir` 和 `embedding_fn`：

```python
engine = RAGEngine(
    persist_dir="chroma_db_bge_m3",
    embedding_fn=get_embedding_fn("bge-m3")
)
```

### Q5：评估集怎么准备？

`eval_set.json` 格式：

```json
[
  {
    "q": "员工年假有多少天？",
    "gold": "入职满一年享5天年假",
    "source": "员工手册_v2.3.md",
    "type": "answerable"
  },
  {
    "q": "公司有没有班车？",
    "gold": "",
    "source": "",
    "type": "unanswerable"
  }
]
```

### Q6：如何重建向量库？

```bash
python build_all_indexes.py --force
```

或直接删除对应目录后重新运行建库脚本。

---

## 🛠️ 技术栈

| 类别 | 组件 |
|:---|:---|
| 大模型 | DeepSeek（`deepseek-flash`） |
| 向量化 | `text-embedding-v4` / `bge-m3` |
| 向量库 | Chroma |
| 检索 | 向量检索 + BM25 + RRF 融合 |
| 重排 | `BAAI/bge-reranker-v2-m3` |
| 框架 | LangChain |
| 前端 | Streamlit |
| 可视化 | Matplotlib |
| 文档加载 | PyMuPDFLoader / TextLoader |

---

## 🗺️ 后续规划

- [x] 流式输出（随模型返回的文本片段逐步显示）
- [x] 对话历史导出（Markdown）
- [x] 多用户会话隔离（按浏览器会话）
- [ ] 文档增量更新（不重建全库）
- [ ] 接入更多向量模型
- [ ] 增加 RAGAS 自动评估

---

## 📄 License

MIT

---

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [Chroma](https://github.com/chroma-core/chroma)
- [DeepSeek](https://platform.deepseek.com)
- [Streamlit](https://streamlit.io)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
