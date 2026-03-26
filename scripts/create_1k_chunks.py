import json, random
from pathlib import Path

INP = Path("data/processed/corpus_10k_chunks.jsonl")
OUT = Path("data/processed/corpus_1k_chunks.jsonl")
TARGET_DOCS = 1000
SEED = 42

doc_ids, seen = [], set()
with INP.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        d = r["id"]
        if d not in seen:
            seen.add(d)
            doc_ids.append(d)

random.Random(SEED).shuffle(doc_ids)
keep = set(doc_ids[:TARGET_DOCS])

OUT.parent.mkdir(parents=True, exist_ok=True)
chunks = 0
with INP.open("r", encoding="utf-8") as fi, OUT.open("w", encoding="utf-8") as fo:
    for line in fi:
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in keep:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            chunks += 1

print("docs:", len(keep))
print("chunks:", chunks)
print("saved:", OUT)

