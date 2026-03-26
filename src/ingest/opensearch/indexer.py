from opensearchpy import OpenSearch, helpers


def ensure_index(client: OpenSearch, index_name: str, recreate: bool = False) -> None:
    if recreate and client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    if client.indices.exists(index=index_name):
        return

    body = {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "id": {"type": "long"},
                "chunk_id": {"type": "keyword"},
                "chunk_text": {"type": "text"},
                "section_type": {"type": "keyword"},
                "article_no": {"type": "integer"},
                "clause_no": {"type": "integer"},
                "document_number": {"type": "keyword"},
                "title": {"type": "text"},
                "url": {"type": "keyword"},
                "legal_type": {"type": "keyword"},
                "legal_sectors": {"type": "keyword"},
                "issuing_authority": {"type": "keyword"},
                "issuance_date": {"type": "date", "format": "dd/MM/yyyy||strict_date_optional_time"},
                "signers": {"type": "keyword"},
                "word_count": {"type": "integer"},
                "token_estimate": {"type": "integer"},
            }
        },
    }
    client.indices.create(index=index_name, body=body)


def bulk_index_chunks(
    client: OpenSearch,
    index_name: str,
    rows,
    batch_size: int = 500,
) -> tuple[int, int]:
    def actions():
        for row in rows:
            yield {
                "_op_type": "index",
                "_index": index_name,
                "_id": row["chunk_id"],
                "_source": row,
            }

    ok, errors = helpers.bulk(client, actions(), chunk_size=batch_size, stats_only=True)
    return ok, errors
