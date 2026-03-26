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
    parser.add_argument("--retriever", choices=["bge", "e5"], default="bge")
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()


    client = get_qdrant_client()

    if args.retriever == "bge":
        collection = args.collection or s.qdrant_collection_bge
        model_name = s.bge_model_name
        query_text = args.q
    elif args.retriever == "e5":
        collection = args.collection or s.qdrant_collection_e5
        model_name = s.e5_model_name
        query_text = args.q if args.q.startswith("query:") else f"query: {args.q}"

    model = SentenceTransformer(model_name, device=args.device, model_kwargs={"use_safetensors": True})
    qvec = model.encode(query_text, normalize_embeddings=True).tolist()

    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection,
            query=qvec,
            limit=args.top_k,
            with_payload=True,
        )
        hits = res.points if hasattr(res, "points") else res
    else:
        hits = client.search(
            collection_name=collection,
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
