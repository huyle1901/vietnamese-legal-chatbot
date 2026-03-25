from src.storage.clients import (
    get_mongo_db,
    get_opensearch_client,
    get_qdrant_client,
    get_redis_client,
)

print("Qdrant:", get_qdrant_client().get_collections())
print("Mongo:", get_mongo_db().name)
print("Redis:", get_redis_client().ping())
print("OpenSearch:", get_opensearch_client().info()["version"]["number"])