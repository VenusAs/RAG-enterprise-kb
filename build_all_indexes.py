"""
为所有向量化模型建库，只跑一次
切分方法统一：chunk_size=500, overlap=50
"""

import os
from rag_engine import RAGEngine, get_embedding_fn

# 要建库的模型列表
MODELS = [
    {"name": "text-embedding-v4", "persist_dir": "chroma_db_v4"},
    {"name": "bge-m3", "persist_dir": "chroma_db_bge_m3"},
]

# 统一切分参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def main():
    print("=" * 70)
    print("为所有向量化模型建库（统一切分方法）")
    print("=" * 70)

    for m in MODELS:
        print(f"\n{'=' * 70}")
        print(f"正在处理：{m['name']}")
        print(f"{'=' * 70}")

        if os.path.exists(m["persist_dir"]) and len(os.listdir(m["persist_dir"])) > 0:
            print(f"✅ 向量库已存在，跳过: {m['persist_dir']}")
            continue

        engine = RAGEngine(
            corpus_dir="corpus",
            persist_dir=m["persist_dir"],
            embedding_fn=get_embedding_fn(m["name"])
        )

        docs = engine.load_documents()
        chunks = engine.split_documents(
            docs,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        engine.build_index(chunks, force_rebuild=True)

    print("\n" + "=" * 70)
    print("✅ 所有模型建库完成")
    print("=" * 70)


if __name__ == "__main__":
    main()