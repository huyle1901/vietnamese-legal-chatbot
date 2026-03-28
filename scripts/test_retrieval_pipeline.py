import argparse

from src.retrieval.pipeline import build_llm_context, build_sources, retrieve_context

q = "hành lang bảo vệ cống qua đê bao nhiêu mét"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, default=q)
    parser.add_argument("--retrieve-k", type=int, default=24)
    parser.add_argument("--final-k", type=int, default=8)
    args = parser.parse_args()

    chunks = retrieve_context(
        query=args.q,
        retrieve_top_k=args.retrieve_k,
        final_top_k=args.final_k,
        bm25_k=30,
        dense_k=30,
        max_chunks_per_doc=2,
    )

    print("=== TOP CHUNKS ===")
    for i, ch in enumerate(chunks, 1):
        p = ch.get("payload", {}) or {}
        print(f"\n#{i} chunk_id={ch.get('chunk_id')} rrf={ch.get('rrf_score')} rerank={ch.get('rerank_score')}")
        print(f"title={p.get('title')}")
        print(f"doc={p.get('document_number')} article={p.get('article_no')} clause={p.get('clause_no')}")
        print(f"text={(p.get('chunk_text') or '')[:250]}...")

    print("\n=== SOURCES ===")
    for s in build_sources(chunks):
        print(s)

    print("\n=== CONTEXT FOR LLM ===")
    print(build_llm_context(chunks)[:3000])


if __name__ == "__main__":
    main()