# src/storage/clients.py
from functools import lru_cache

from pymongo import MongoClient
from qdrant_client import QdrantClient
from redis import Redis
from opensearchpy import OpenSearch

from src.config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url)


@lru_cache
def get_mongo_client() -> MongoClient:
    s = get_settings()
    return MongoClient(s.mongo_url)


def get_mongo_db():
    s = get_settings()
    return get_mongo_client()[s.mongo_db_name]


@lru_cache
def get_redis_client() -> Redis:
    s = get_settings()
    return Redis.from_url(s.redis_url, decode_responses=True)


@lru_cache
def get_opensearch_client() -> OpenSearch:
    s = get_settings()
    return OpenSearch(
        hosts=[s.opensearch_url],
        use_ssl=s.opensearch_url.startswith("https"),
        verify_certs=False,  # local dev
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
