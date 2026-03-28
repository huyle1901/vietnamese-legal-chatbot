# src/retrieval/pipeline.py
from __future__ import annotations

from typing import Any

from src.retrieval.hybrid_retrieve import hybrid_retrieve
from src.retrieval.reranker import rerank_candidates


def _limit_chunks_per_document(
    items: list[dict[str, Any]],
    max_chunks_per_doc: int = 2,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    doc_count: dict[str, int] = {}

    for it in items:
        payload = it.get("payload", {}) or {}
        doc_key = str(payload.get("document_number") or payload.get("id") or "")
        if not doc_key:
            kept.append(it)
            continue

        c = doc_count.get(doc_key, 0)
        if c >= max_chunks_per_doc:
            continue
        doc_count[doc_key] = c + 1
        kept.append(it)

    return kept


def retrieve_context(
    query: str,
    retrieve_top_k: int = 24,
    final_top_k: int = 8,
    bm25_k: int = 30,
    dense_k: int = 30,
    max_chunks_per_doc: int = 2,
) -> list[dict[str, Any]]:
    # 1) Hybrid retrieve
    candidates = hybrid_retrieve(
        query=query,
        top_k=retrieve_top_k,
        bm25_k=bm25_k,
        dense_k=dense_k,
    )

    # 2) Rerank (rerank all candidates first)
    ranked = rerank_candidates(query=query, candidates=candidates, top_k=retrieve_top_k)

    # 3) Limit too many chunks from same document
    limited = _limit_chunks_per_document(ranked, max_chunks_per_doc=max_chunks_per_doc)

    return limited[:final_top_k]


def build_llm_context(chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, ch in enumerate(chunks, 1):
        p = ch.get("payload", {}) or {}
        lines.append(
            f"[{i}] title: {p.get('title')}\n"
            f"document_number: {p.get('document_number')} | article: {p.get('article_no')} | clause: {p.get('clause_no')}\n"
            f"url: {p.get('url')}\n"
            f"content: {p.get('chunk_text')}\n"
        )
    return "\n".join(lines)


def build_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ch in chunks:
        p = ch.get("payload", {}) or {}
        out.append(
            {
                "chunk_id": ch.get("chunk_id"),
                "title": p.get("title"),
                "document_number": p.get("document_number"),
                "article_no": p.get("article_no"),
                "clause_no": p.get("clause_no"),
                "url": p.get("url"),
                "rrf_score": ch.get("rrf_score"),
                "rerank_score": ch.get("rerank_score"),
                "sources": ch.get("sources", []),
            }
        )
    return out
