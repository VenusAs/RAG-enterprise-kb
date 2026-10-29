"""
RAG 评估脚本（幻觉率修正版）
- 幻觉率统计所有编造情况（有答案答错 + 无答案硬编）
- 分类型统计：有答案类幻觉率、无答案类幻觉率、总幻觉率
- 所有字符串匹配统一归一化
- 向量化模型对比：只保留 v4 和 bge-m3
"""

import json
import time
import os
from rag_engine import RAGEngine, get_embedding_fn


# ============================================================
# 工具函数
# ============================================================

def load_eval_set(path="eval_set.json"):
    """加载评估集"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text):
    """归一化：消除空格、全角半角、标点差异"""
    if not text:
        return ""
    # 去除所有空白字符
    text = text.replace(" ", "").replace("　", "")
    text = text.replace("\t", "").replace("\n", "").replace("\r", "")
    # 全角标点 → 半角
    text = text.replace("：", ":").replace("，", ",")
    text = text.replace("。", ".").replace("；", ";")
    text = text.replace("！", "!").replace("？", "?")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("【", "[").replace("】", "]")
    # 全角数字/字母 → 半角
    for i in range(10):
        text = text.replace(chr(0xFF10 + i), str(i))
    for i in range(26):
        text = text.replace(chr(0xFF21 + i), chr(65 + i))
    # 全角符号 → 半角
    text = text.replace("％", "%").replace("＃", "#")
    text = text.replace("＆", "&").replace("＊", "*")
    text = text.replace("＋", "+").replace("－", "-")
    text = text.replace("＝", "=").replace("／", "/")
    return text


def is_rejection(answer):
    """判断回答是否为'拒答'"""
    keywords = ["资料中没有提到", "没有找到", "未找到", "无法回答", "没有相关信息", "无法提供"]
    answer_norm = normalize(answer)
    return any(normalize(kw) in answer_norm for kw in keywords)


# ============================================================
# 核心评估函数
# ============================================================

def evaluate_strategy(engine, eval_set, strategy_name, retrieve_fn,
                      generate_fn=None, top_k=3, cost_per_million=0.07):
    """
    评估单个策略
    幻觉率统计所有编造情况
    """
    if generate_fn is None:
        generate_fn = engine.generate

    # 检索指标
    recall_hit = 0
    mrr_total = 0.0
    citation_hit = 0

    # 生成指标（有答案类）
    answerable_total = 0
    correct_answer = 0
    wrong_answer = 0
    missed_answer = 0

    # 生成指标（无答案类）
    reject_total = 0
    reject_correct = 0

    # 幻觉指标
    hallucination_answerable = 0
    hallucination_unanswerable = 0

    # 性能指标
    total_time = 0
    total_tokens = 0
    n = len(eval_set)

    for item in eval_set:
        query = item["q"]
        gold = item["gold"]
        source = item["source"]
        qtype = item.get("type", "answerable")

        # 检索
        start = time.time()
        docs = retrieve_fn(query)
        elapsed = time.time() - start
        total_time += elapsed

        docs = (docs or [])[:top_k]
        contents = [doc.page_content for doc in docs]
        sources = [doc.metadata.get("source", "") for doc in docs]
        total_tokens += sum(len(c) * 1.5 for c in contents)

        # 归一化
        gold_norm = normalize(gold)
        contents_norm = [normalize(c) for c in contents]

        # 生成回答
        context = "\n\n".join(contents)
        answer = generate_fn(query, context)
        answer_norm = normalize(answer)

        # ---------- 有答案类问题 ----------
        if qtype == "answerable":
            answerable_total += 1

            # 检索指标
            hit = any(gold_norm in c for c in contents_norm)
            recall_hit += 1 if hit else 0

            rank = 0
            for j, c in enumerate(contents_norm, 1):
                if gold_norm in c:
                    rank = j
                    break
            mrr_total += (1.0 / rank) if rank else 0.0

            source_norm = normalize(source)
            sources_norm = [normalize(s) for s in sources]
            citation_hit += 1 if any(source_norm in s for s in sources_norm) else 0

            # 生成指标
            if is_rejection(answer):
                missed_answer += 1
            elif gold_norm in answer_norm:
                correct_answer += 1
            else:
                wrong_answer += 1
                hallucination_answerable += 1

        # ---------- 无答案类问题 ----------
        else:
            reject_total += 1
            if is_rejection(answer):
                reject_correct += 1
            else:
                hallucination_unanswerable += 1

    # 汇总
    a = max(answerable_total, 1)
    r = max(reject_total, 1)
    total_hallucination = hallucination_answerable + hallucination_unanswerable
    cost_usd = total_tokens / 1_000_000 * cost_per_million

    return {
        "策略": strategy_name,
        "Recall@3": f"{recall_hit / a:.2%}",
        "MRR": f"{mrr_total / a:.4f}",
        "引用准确率": f"{citation_hit / a:.2%}",
        "答案正确率": f"{correct_answer / a:.2%}",
        "漏答率": f"{missed_answer / a:.2%}",
        "有答案类幻觉率": f"{hallucination_answerable / a:.2%}",
        "无答案类幻觉率": f"{hallucination_unanswerable / r:.2%}",
        "总幻觉率": f"{total_hallucination / n:.2%}",
        "拒答准确率": f"{reject_correct / r:.2%}",
        "平均耗时": f"{total_time / n:.3f}s",
        "成本": f"${cost_usd:.4f}" if cost_usd > 0 else "免费",
    }


# ============================================================
# 实验1：四种召回策略对比
# ============================================================

def experiment_recall_strategies(engine, eval_set):
    print("\n" + "=" * 70)
    print("实验 1：四种召回策略对比")
    print("=" * 70)

    def retrieve_vector(q):
        return engine.vector_store.similarity_search(q, k=3)

    def retrieve_bm25(q):
        return engine.bm25_retriever.invoke(q)[:3]

    def retrieve_ensemble(q):
        return engine.ensemble_retriever.invoke(q)[:3]

    def retrieve_rerank(q):
        return engine.retrieval_chain.invoke(q)

    strategies = [
        ("纯向量检索", retrieve_vector),
        ("纯 BM25 检索", retrieve_bm25),
        ("混合检索（向量+BM25）", retrieve_ensemble),
        ("混合检索 + 重排", retrieve_rerank),
    ]

    results = []
    for name, fn in strategies:
        print(f"\n正在评估：{name}")
        result = evaluate_strategy(engine, eval_set, name, fn)
        results.append(result)
        for k, v in result.items():
            if k != "策略":
                print(f"  {k}: {v}")
    return results


# ============================================================
# 实验2：切分方法对比（仅用 bge-m3）
# ============================================================

def experiment_chunking_methods(eval_set):
    print("\n" + "=" * 70)
    print("实验 2：切分方法对比（仅用 bge-m3）")
    print("=" * 70)

    strategies = [
        ("chunk_size=200, overlap=0", 200, 0, ["\n\n", "\n", "。", "！", "？", "，", " ", ""]),
        ("chunk_size=500, overlap=50", 500, 50, ["\n\n", "\n", "。", "！", "？", "，", " ", ""]),
        ("chunk_size=1000, overlap=100", 1000, 100, ["\n\n", "\n", "。", "！", "？", "，", " ", ""]),
        ("按句号切（chunk_size=200）", 200, 0, ["。", "！", "？"]),
        ("按段落切（chunk_size=1000）", 1000, 0, ["\n\n"]),
    ]

    results = []
    for name, chunk_size, overlap, seps in strategies:
        print(f"\n正在评估：{name}")

        safe_name = name.replace("=", "").replace(",", "").replace(" ", "_").replace("（", "").replace("）", "")
        persist_dir = f"chroma_db_bge_m3_{safe_name}"

        engine = RAGEngine(
            corpus_dir="corpus",
            persist_dir=persist_dir,
            embedding_fn=get_embedding_fn("bge-m3")
        )

        # 库存在则加载，不存在则建库
        if engine.vector_store_exists():
            print(f"✅ 向量库已存在，加载: {persist_dir}")
            engine.load_existing_index()
        else:
            print(f"🔨 向量库不存在，建库: {persist_dir}")
            docs = engine.load_documents()
            chunks = engine.split_documents(
                docs, chunk_size=chunk_size, chunk_overlap=overlap,
                separators=seps, use_markdown_splitter=False
            )
            engine.build_index(chunks, force_rebuild=True)

        if engine.retrieval_chain is None:
            print(f"❌ {name} 检索链构建失败，跳过")
            continue

        def retrieve(q):
            return engine.retrieval_chain.invoke(q)

        result = evaluate_strategy(engine, eval_set, name, retrieve, cost_per_million=0.0)
        results.append(result)
        for k, v in result.items():
            if k != "策略":
                print(f"  {k}: {v}")
    return results


# ============================================================
# 实验3：向量化模型对比（只保留 v4 和 bge-m3）
# ============================================================

def experiment_embedding_models(eval_set):
    print("\n" + "=" * 70)
    print("实验 3：向量化模型对比（v4 vs bge-m3）")
    print("=" * 70)

    models = [
        {"name": "text-embedding-v4", "persist_dir": "chroma_db_v4", "cost": 0.07},
        {"name": "bge-m3", "persist_dir": "chroma_db_bge_m3", "cost": 0.0},
    ]

    results = []
    for m in models:
        if not os.path.exists(m["persist_dir"]):
            print(f"⚠️ {m['name']} 向量库不存在，请先跑 build_all_indexes.py")
            continue

        print(f"\n正在评估：{m['name']}")
        engine = RAGEngine(
            persist_dir=m["persist_dir"],
            embedding_fn=get_embedding_fn(m["name"])
        )
        engine.load_existing_index()

        def retrieve(q):
            return engine.retrieval_chain.invoke(q)

        result = evaluate_strategy(engine, eval_set, m["name"], retrieve, cost_per_million=m["cost"])
        results.append(result)
        for k, v in result.items():
            if k != "策略":
                print(f"  {k}: {v}")
    return results


# ============================================================
# 实验4：防幻觉 Prompt 对比
# ============================================================

def experiment_anti_hallucination(engine, eval_set):
    print("\n" + "=" * 70)
    print("实验 4：防幻觉 Prompt 对比")
    print("=" * 70)

    def generate_no_constraint(query, context):
        prompt = f"""请根据以下资料回答问题。

资料：
{context}

问题：{query}

回答："""
        return engine.llm.invoke(prompt).content

    def generate_with_constraint(query, context):
        prompt = f"""请根据以下资料回答问题。

规则：
1. 只使用资料中的信息，不要编造。
2. 如果资料中没有相关信息，直接说"资料中没有提到"。
3. 不要尝试从资料中推测答案。
4. 如果资料中的内容与问题不直接相关，也要说"资料中没有提到"。
5. 回答末尾必须标注来源，格式：[来源：文档名]

资料：
{context}

问题：{query}

回答："""
        return engine.llm.invoke(prompt).content

    strategies = [
        ("无约束 Prompt", generate_no_constraint),
        ("RAG + 引用 + 拒答约束", generate_with_constraint),
    ]

    results = []
    for name, gen_fn in strategies:
        print(f"\n正在评估：{name}")

        def retrieve(q):
            return engine.retrieval_chain.invoke(q)

        result = evaluate_strategy(engine, eval_set, name, retrieve, generate_fn=gen_fn)
        results.append(result)
        for k, v in result.items():
            if k != "策略":
                print(f"  {k}: {v}")
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("RAG 评估 · 四组对比实验（幻觉率修正版）")
    print("=" * 70)

    eval_set = load_eval_set()
    print(f"\n评估集共 {len(eval_set)} 条")
    print(f"  有答案：{sum(1 for x in eval_set if x.get('type') == 'answerable')} 条")
    print(f"  无答案：{sum(1 for x in eval_set if x.get('type') == 'unanswerable')} 条")

    all_results = {}

    engine_v4 = RAGEngine(
        persist_dir="chroma_db_v4",
        embedding_fn=get_embedding_fn("text-embedding-v4")
    )
    if not engine_v4.vector_store_exists():
        print("❌ chroma_db_v4 不存在，请先跑 build_all_indexes.py")
        return
    engine_v4.load_existing_index()

    # 实验1
    all_results["召回策略"] = experiment_recall_strategies(engine_v4, eval_set)

    # 实验2
    all_results["切分方法"] = experiment_chunking_methods(eval_set)

    # 实验3
    all_results["向量化模型"] = experiment_embedding_models(eval_set)

    # 实验4
    all_results["防幻觉"] = experiment_anti_hallucination(engine_v4, eval_set)

    # 保存
    with open("evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n✅ 所有结果已保存到 evaluation_result.json")


if __name__ == "__main__":
    main()