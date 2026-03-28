from functools import lru_cache
from sentence_transformers import CrossEncoder
import torch

from src.config import get_settings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache
def _get_reranker():
    s = get_settings()
    return CrossEncoder(
        s.local_reranker_model,
        device=DEVICE,
        trust_remote_code=True,
    )

def rerank_candidates(query: str, candidates: list[dict], top_k: int = 8) -> list[dict]:
    if not candidates:
        return []

    pairs = []
    for c in candidates:
        text = (c.get("payload", {}) or {}).get("chunk_text", "")
        pairs.append((query, text))

    scores = _get_reranker().predict(pairs, batch_size=16, show_progress_bar=False)

    for c, sc in zip(candidates, scores):
        c["rerank_score"] = float(sc)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]