from pathlib import Path
from datasets import load_dataset

OUT = Path("data/raw/metadata_sample_10k.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "th1nhng0/vietnamese-legal-documents",
    "metadata",
    split="data[:10000]",
)

print("rows:", len(ds))
print("columns:", ds.column_names)

ds.to_json(str(OUT), orient="records", lines=True, force_ascii=False)
print("saved:", OUT)