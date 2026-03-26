import argparse
import hashlib
from pathlib import Path

import torch
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import get_settings
from src.ingest.common.jsonl_io import read_jsonl
from src.storage.clients import get_qdrant_client


def chunk_id_to_int(chunk_id: str) -> int:
    h = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="big", signed=False)


def batched(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_collection(client, collection_name: str, vector_size: int, recreate: bool = False):
    exists = any(c.name == collection_name for c in client.get_collections().collections)
    if recreate and exists:
        client.delete_collection(collection_name=collection_name)
        exists = False

    if exists:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def main():
    s = get_settings()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-size", choices=["1k", "10k"], default="1k")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--recreate-collection", action="store_true")
    parser.add_argument("--collection", type=str, default=s.qdrant_collection_e5)
    parser.add_argument("--model", type=str, default=s.e5_model_name)  # intfloat/multilingual-e5-large
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    client = get_qdrant_client()
    model = SentenceTransformer(args.model, device=args.device, model_kwargs={"use_safetensors": True})

    test_vec = model.encode(["query: test"], normalize_embeddings=True)
    vector_size = len(test_vec[0])
    ensure_collection(client, args.collection, vector_size, recreate=args.recreate_collection)

    chunks_input_1k = getattr(s, "chunks_input_1k", "data/processed/corpus_1k_chunks.jsonl")
    chunks_input_10k = getattr(s, "chunks_input_10k", "data/processed/corpus_10k_chunks.jsonl")

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = Path(chunks_input_1k if args.dataset_size == "1k" else chunks_input_10k)

    rows = read_jsonl(input_path)
    print(f"Input: {input_path}")
    print(f"Device: {args.device}")
    total = 0

    for batch in tqdm(batched(rows, args.batch_size), desc="Ingest E5", unit="batch"):
        texts = [f"passage: {r['chunk_text']}" for r in batch]
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )

        points = []
        for r, vec in zip(batch, vectors):
            payload = {
                "id": r["id"],
                "chunk_id": r["chunk_id"],
                "chunk_text": r["chunk_text"],
                "section_type": r.get("section_type"),
                "article_no": r.get("article_no"),
                "clause_no": r.get("clause_no"),
                "document_number": r.get("document_number"),
                "title": r.get("title"),
                "url": r.get("url"),
                "legal_type": r.get("legal_type"),
                "legal_sectors": r.get("legal_sectors"),
                "issuing_authority": r.get("issuing_authority"),
                "issuance_date": r.get("issuance_date"),
                "signers": r.get("signers"),
            }
            points.append(
                models.PointStruct(
                    id=chunk_id_to_int(r["chunk_id"]),
                    vector=vec.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=args.collection, points=points, wait=True)
        total += len(points)

        if total % 5000 == 0:
            print(f"Upserted {total} points...")

    print(f"Done. Total upserted: {total}, collection: {args.collection}")


if __name__ == "__main__":
    main()
