from src.storage.clients import get_opensearch_client
from src.config import get_settings

q = "hành lang bảo vệ cống qua đê bao nhiêu mét"

s = get_settings()
client = get_opensearch_client()

body = {
    "size": 5,
    "query": {
        "multi_match": {
            "query": q,
            "fields": ["chunk_text^3", "title^2", "document_number", "legal_type", "legal_sectors"],
            "type": "best_fields"
        }
    }
}

res = client.search(index=s.opensearch_index, body=body)

for i, hit in enumerate(res["hits"]["hits"], 1):
    src = hit["_source"]
    print(f"\n#{i} score={hit['_score']:.4f} chunk_id={src['chunk_id']}")
    print(f"title={src.get('title')}")
    print(f"text={src.get('chunk_text','')[:300]}...")