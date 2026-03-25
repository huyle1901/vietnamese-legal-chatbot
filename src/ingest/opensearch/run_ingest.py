import argparse
from pathlib import Path

from src.config import get_settings
from src.ingest.common.jsonl_io import read_jsonl
from src.ingest.opensearch.indexer import ensure_index, bulk_index_chunks
from src.storage.clients import get_opensearch_client


def main():
    s = get_settings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/corpus_10k_chunks.jsonl")
    parser.add_argument("--index", type=str, default=s.opensearch_index)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--recreate-index", action="store_true")
    args = parser.parse_args()

    client = get_opensearch_client()
    ensure_index(client, args.index, recreate=args.recreate_index)

    rows = read_jsonl(Path(args.input))
    ok, errors = bulk_index_chunks(client, args.index, rows, batch_size=args.batch_size)

    print(f"Indexed docs: {ok}, errors: {errors}, index: {args.index}")


if __name__ == "__main__":
    main()
