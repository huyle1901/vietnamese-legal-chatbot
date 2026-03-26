import argparse
from pathlib import Path

from src.config import get_settings
from src.ingest.common.jsonl_io import read_jsonl
from src.ingest.opensearch.indexer import ensure_index, bulk_index_chunks
from src.storage.clients import get_opensearch_client


def resolve_input_path(
    input_arg: str | None,
    dataset_size: str | None,
    chunks_input_1k: str,
    chunks_input_10k: str,
    max_docs_for_ingest: int,
) -> Path:
    # 1) user truyền --input thì dùng luôn
    if input_arg:
        return Path(input_arg)

    # 2) user chọn --dataset-size
    if dataset_size == "1k":
        return Path(chunks_input_1k)
    if dataset_size == "10k":
        return Path(chunks_input_10k)

    # 3) fallback theo config MAX_DOCS_FOR_INGEST
    if max_docs_for_ingest and max_docs_for_ingest <= 1000:
        return Path(chunks_input_1k)
    return Path(chunks_input_10k)


def main():
    s = get_settings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--dataset-size", choices=["1k", "10k"], default=None)
    parser.add_argument("--index", type=str, default=s.opensearch_index)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--recreate-index", action="store_true")
    args = parser.parse_args()

    input_path = resolve_input_path(
        input_arg=args.input,
        dataset_size=args.dataset_size,
        chunks_input_1k=s.chunks_input_1k,
        chunks_input_10k=s.chunks_input_10k,
        max_docs_for_ingest=s.max_docs_for_ingest,
    )

    client = get_opensearch_client()
    ensure_index(client, args.index, recreate=args.recreate_index)

    rows = read_jsonl(input_path)
    ok, errors = bulk_index_chunks(client, args.index, rows, batch_size=args.batch_size)

    print(f"input={input_path}")
    print(f"Indexed docs: {ok}, errors: {errors}, index: {args.index}")


if __name__ == "__main__":
    main()
