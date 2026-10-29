"""
企业知识库问答系统 · Streamlit 前端
支持：
- 自定义召回策略
- 可调 top_n（送给 LLM 的文档数）
- 可调召回候选数 k
- 带上下文记忆的多轮对话
- 展示检索来源和原文片段
"""

import os
import streamlit as st

from rag_engine import RAGEngine, get_embedding_fn


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="企业知识库问答系统",
    page_icon="📚",
    layout="wide"
)

st.title("📚 企业知识库问答系统")
st.caption("基于混合检索 + Cross-Encoder 重排的 RAG 问答")


# ============================================================
# 侧边栏配置
# ============================================================
with st.sidebar:
    st.header("⚙️ 检索配置")

    # 召回策略
    recall_strategy = st.selectbox(
        "召回策略",
        options=[
            "纯向量检索",
            "纯 BM25 检索",
            "混合检索（向量+BM25）",
            "混合检索 + 重排"
        ],
        index=3
    )

    # top_n：最终送给 LLM 的文档数
    top_n = st.slider(
        "top_n（送给 LLM 的文档数）",
        min_value=1, max_value=10, value=3, step=1
    )

    # k：召回候选数
    recall_k = st.slider(
        "召回候选数 k",
        min_value=5, max_value=30, value=20, step=5,
        help="混合检索时每路召回的候选数"
    )

    st.divider()

    # 清空对话
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# 加载 RAG 引擎（只初始化一次）
# ============================================================
@st.cache_resource(show_spinner="正在加载 RAG 引擎...")
def load_engine():
    engine = RAGEngine(
        corpus_dir="corpus",
        persist_dir="chroma_db_v4",
        embedding_fn=get_embedding_fn("text-embedding-v4")
    )
    if engine.vector_store_exists():
        engine.load_existing_index()
    else:
        docs = engine.load_documents()
        chunks = engine.split_documents(docs)
        engine.build_index(chunks, force_rebuild=True)
    return engine


try:
    engine = load_engine()
    st.sidebar.success("✅ 向量库已加载：chroma_db_v4")
except Exception as e:
    st.error(f"❌ 引擎加载失败：{e}")
    st.stop()


# ============================================================
# 根据侧边栏选择，动态切换检索器
# ============================================================
def get_retriever(strategy, top_n, recall_k):
    """根据策略名返回检索函数"""
    if strategy == "纯向量检索":
        def fn(q):
            return engine.vector_store.similarity_search(q, k=top_n)
        return fn

    if strategy == "纯 BM25 检索":
        def fn(q):
            return engine.bm25_retriever.invoke(q)[:top_n]
        return fn

    if strategy == "混合检索（向量+BM25）":
        engine.vector_retriever.search_kwargs = {"k": recall_k}
        engine.bm25_retriever.k = recall_k
        def fn(q):
            return engine.ensemble_retriever.invoke(q)[:top_n]
        return fn

    if strategy == "混合检索 + 重排":
        engine.vector_retriever.search_kwargs = {"k": recall_k}
        engine.bm25_retriever.k = recall_k
        engine.reranker.top_n = top_n
        def fn(q):
            return engine.retrieval_chain.invoke(q)
        return fn

    # 默认走完整检索链
    def fn(q):
        return engine.retrieval_chain.invoke(q)
    return fn


retrieve_fn = get_retriever(recall_strategy, top_n, recall_k)


# ============================================================
# 上下文拼接
# ============================================================
def build_context(docs):
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "未知")
        source_name = os.path.basename(source)

        h1 = doc.metadata.get("h1")
        h2 = doc.metadata.get("h2")
        h3 = doc.metadata.get("h3")

        if h3:
            location = f"{h1} → {h2} → {h3}"
        elif h2:
            location = f"{h1} → {h2}"
        elif h1:
            location = h1
        else:
            location = ""

        if location:
            parts.append(f"[来源：{source_name} {location}]\n{doc.page_content}")
        else:
            parts.append(f"[来源：{source_name}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ============================================================
# 生成回答（带上下文记忆）
# ============================================================
def generate_with_memory(question, context, history):
    """
    把最近几轮对话 + 当前检索上下文一起交给 LLM
    history: List[dict]，每个 dict 形如 {"role": "user"/"assistant", "content": "..."}
    """
    # 只保留最近 3 轮，避免 prompt 过长
    recent = history[-6:]

    history_text = ""
    for msg in recent:
        role = "用户" if msg["role"] == "user" else "助手"
        history_text += f"{role}：{msg['content']}\n"

    prompt = f"""你是一个企业知识库助手，请根据以下资料回答用户问题。

规则：
1. 只使用资料中的信息，不要编造。
2. 如果资料中没有相关信息，直接说“资料中没有提到”。
3. 回答末尾必须标注来源，格式：[来源：文档名 章节名]
4. 可以结合对话历史理解用户的指代（如“它”“这个”），但答案必须来自资料。

对话历史：
{history_text if history_text else "（无）"}

资料：
{context}

当前问题：{question}

回答：
"""
    response = engine.llm.invoke(prompt)
    return response.content


# ============================================================
# 对话主循环
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
question = st.chat_input("请输入你的问题...")

if question:
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. 检索
    with st.spinner("正在检索..."):
        try:
            docs = retrieve_fn(question)
            docs = (docs or [])[:top_n]
        except Exception as e:
            st.error(f"检索失败：{e}")
            docs = []

    # 3. 生成
    with st.chat_message("assistant"):
        if not docs:
            answer = "没有找到相关文档"
            st.markdown(answer)
        else:
            context = build_context(docs)

            with st.spinner("正在生成回答..."):
                answer = generate_with_memory(
                    question, context, st.session_state.messages[:-1]
                )

            st.markdown(answer)

            # 4. 展示来源
            with st.expander("📎 查看检索到的原文片段"):
                for i, doc in enumerate(docs, 1):
                    source = os.path.basename(doc.metadata.get("source", "未知"))
                    h1 = doc.metadata.get("h1", "")
                    h2 = doc.metadata.get("h2", "")
                    h3 = doc.metadata.get("h3", "")
                    loc = " → ".join([x for x in [h1, h2, h3] if x])
                    st.markdown(f"**{i}. {source}** {loc}")
                    st.text(
                        doc.page_content[:400]
                        + ("..." if len(doc.page_content) > 400 else "")
                    )
                    st.divider()

    # 5. 记录助手消息
    st.session_state.messages.append({"role": "assistant", "content": answer})