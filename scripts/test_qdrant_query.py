# scripts/test_qdrant_query.py
import argparse
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.storage.clients import get_qdrant_client

q = "hành lang bảo vệ cống qua đê bao nhiêu mét"

def main():
    s = get_settings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, default=q, required=False)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", type=str, default=s.qdrant_collection_bge)
    parser.add_argument("--model", type=str, default=s.bge_model_name)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = SentenceTransformer(args.model, device=args.device, model_kwargs={"use_safetensors": True})
    qvec = model.encode([args.q], normalize_embeddings=True)[0].tolist()

    client = get_qdrant_client()

    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=args.collection,
            query=qvec,
            limit=args.top_k,
            with_payload=True,
        )
        hits = res.points if hasattr(res, "points") else res
    else:
        hits = client.search(
            collection_name=args.collection,
            query_vector=qvec,
            limit=args.top_k,
            with_payload=True,
        )

    for i, h in enumerate(hits, 1):
        if isinstance(h, tuple):
            h = h[0]

        p = getattr(h, "payload", {}) or {}
        score = getattr(h, "score", 0.0)

        print(f"\n#{i} score={score:.4f} chunk_id={p.get('chunk_id')}")
        print(f"title={p.get('title')}")
        print(f"text={(p.get('chunk_text') or '')[:300]}...")




if __name__ == "__main__":
    main()
