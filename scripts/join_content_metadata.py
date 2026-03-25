import json
from pathlib import Path

CONTENT_PATH = Path("data/raw/content_sample_10k.jsonl")
METADATA_PATH = Path("data/raw/metadata_sample_10k.jsonl")
OUT_PATH = Path("data/processed/corpus_10k.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# Load metadata -> map by id
metadata_map = {}
for row in read_jsonl(METADATA_PATH):
    rid = int(row["id"])
    metadata_map[rid] = row

matched = 0
missing_meta = 0

with OUT_PATH.open("w", encoding="utf-8") as out:
    for row in read_jsonl(CONTENT_PATH):
        rid = int(row["id"])
        meta = metadata_map.get(rid)

        if meta is None:
            missing_meta += 1
            continue

        merged = {
            "id": rid,
            "content": row.get("content", ""),
            "document_number": meta.get("document_number"),
            "title": meta.get("title"),
            "url": meta.get("url"),
            "legal_type": meta.get("legal_type"),
            "legal_sectors": meta.get("legal_sectors"),
            "issuing_authority": meta.get("issuing_authority"),
            "issuance_date": meta.get("issuance_date"),
            "signers": meta.get("signers"),
        }

        out.write(json.dumps(merged, ensure_ascii=False) + "\n")
        matched += 1

print(f"matched={matched}, missing_meta={missing_meta}, out={OUT_PATH}")