from __future__ import annotations

from functools import lru_cache
from typing import Any

from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.storage.clients import get_opensearch_client, get_qdrant_client

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache
def _get_bge():
    s = get_settings()
    return SentenceTransformer(s.bge_model_name, device=DEVICE, model_kwargs={"use_safetensors": True})

@lru_cache
def _get_e5():
    s = get_settings()
    return SentenceTransformer(s.e5_model_name, device=DEVICE, model_kwargs={"use_safetensors": True})


def _search_opensearch(query: str, top_k: int) -> list[dict[str, Any]]:
    s = get_settings()
    client = get_opensearch_client()
    body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["chunk_text^3", "title^2", "document_number", "legal_type", "legal_sectors"],
                "type": "best_fields",
            }
        },
    }
    res = client.search(index=s.opensearch_index, body=body)
    hits = []
    for h in res["hits"]["hits"]:
        src = h["_source"]
        hits.append(
            {
                "chunk_id": src.get("chunk_id"),
                "score": float(h["_score"]),
                "payload": src,
                "retriever": "bm25",
            }
        )
    return hits

def _search_qdrant(collection: str, qvec: list[float], top_k: int, retriever_name: str) -> list[dict[str, Any]]:
    client = get_qdrant_client()

    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection,
            query=qvec,
            limit=top_k,
            with_payload=True,
        )
        points = res.points if hasattr(res, "points") else res
    else:
        points = client.search(
            collection_name=collection,
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
        )

    out = []
    for p in points:
        if isinstance(p, tuple):
            p = p[0]
        payload = getattr(p, "payload", {}) or {}
        out.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "score": float(getattr(p, "score", 0.0)),
                "payload": payload,
                "retriever": retriever_name,
            }
        )
    return out

def _rrf_fuse(result_lists: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    # Reciprocal Rank Fusion: robust khi score mỗi retriever khác thang đo
    merged: dict[str, dict[str, Any]] = {}

    for lst in result_lists:
        for rank, item in enumerate(lst, start=1):
            cid = item.get("chunk_id")
            if not cid:
                continue
            rrf = 1.0 / (k + rank)

            if cid not in merged:
                merged[cid] = {
                    "chunk_id": cid,
                    "payload": item.get("payload", {}),
                    "rrf_score": 0.0,
                    "sources": set(),
                }

            merged[cid]["rrf_score"] += rrf
            merged[cid]["sources"].add(item.get("retriever"))

    fused = list(merged.values())
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    for x in fused:
        x["sources"] = sorted(list(x["sources"]))

    return fused

def hybrid_retrieve(query: str, 
                    top_k: int = 10, 
                    bm25_k: int=30, 
                    dense_k: int = 30) -> list[dict[str, Any]]:
    
    s = get_settings()

    # BM25
    bm25_hits = _search_opensearch(query, top_k=bm25_k)

    # BGE
    bge_vec = _get_bge().encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
    bge_hits = _search_qdrant(s.qdrant_collection_bge, qvec=bge_vec, top_k=dense_k, retriever_name="bge")

    # E5
    e5_query = query if query.startswith("query: ") else f"query: {query}"
    e5_vec = _get_e5().encode([e5_query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
    e5_hits = _search_qdrant(s.qdrant_collection_e5, qvec=e5_vec, top_k=dense_k, retriever_name="e5")

    # Fuse results
    fused = _rrf_fuse([bm25_hits, bge_hits, e5_hits], k=60)
    return fused[:top_k]

if __name__ == "__main__":
    import argparse
    import json

    q = "hành lang bảo vệ cống qua đê bao nhiêu mét"

    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, default=q)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-k", type=int, default=30)
    parser.add_argument("--dense-k", type=int, default=30)
    args = parser.parse_args()



    results = hybrid_retrieve(
        query=args.q,
        top_k=args.top_k,
        bm25_k=args.bm25_k,
        dense_k=args.dense_k,
    )

    for i, r in enumerate(results, 1):
        p = r.get("payload", {})
        print(f"\n#{i} rrf={r.get('rrf_score', 0):.6f} sources={r.get('sources', [])} chunk_id={r.get('chunk_id')}")
        print(f"title={p.get('title')}")
        print(f"doc={p.get('document_number')} article={p.get('article_no')} clause={p.get('clause_no')}")
        print(f"text={(p.get('chunk_text') or '')[:300]}...")

    


