from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "vietlaw-rag-v2"
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    app_port: int = Field(default=8000, alias="APP_PORT")

    # Database / Storage
    mongo_url: str = Field(default="mongodb://localhost:27017", alias="MONGO_URL")
    mongo_db_name: str = Field(default="vietlaw_rag", alias="MONGO_DB_NAME")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection_bge: str = Field(default="law_bge", alias="QDRANT_COLLECTION_BGE")
    qdrant_collection_e5: str = Field(default="law_e5", alias="QDRANT_COLLECTION_E5")

    opensearch_url: str = Field(default="http://localhost:9200", alias="OPENSEARCH_URL")
    opensearch_index: str = Field(default="law_bm25", alias="OPENSEARCH_INDEX")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Embedding models
    bge_model_name: str = Field(default="BAAI/bge-m3", alias="BGE_MODEL_NAME")
    e5_model_name: str = Field(default="intfloat/multilingual-e5-large", alias="E5_MODEL_NAME")

    # Retrieval / Rerank
    retriever_top_k: int = Field(default=30, alias="RETRIEVER_TOP_K")
    reranker_top_k: int = Field(default=8, alias="RERANKER_TOP_K")
    rerank_provider: str = Field(default="local", alias="RERANK_PROVIDER")  # local, cohere, openai(llm)

    # Local reranker
    local_reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="LOCAL_RERANKER_MODEL")

    # Cohere reranker
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    cohere_rerank_model: str = Field(default="rerank-v3.5", alias="COHERE_RERANK_MODEL")

    # Generation model (LLM answer)
    answer_provider: str = Field(default="local", alias="ANSWER_MODEL_PROVIDER")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5.2-mini", alias="OPENAI_MODEL_NAME")

    # Local model generation
    local_llm_base_url: str = Field(default="http://localhost:8001", alias="LOCAL_LLM_BASE_URL")
    local_llm_model_name: str = Field(default="Qwen/Qwen2.5-3B-Instruct", alias="LOCAL_LLM_MODEL_NAME")

    # Ingest control
    max_docs_for_ingest: int = Field(default=1000, alias="MAX_DOCS_FOR_INGEST")  # 0 = full
    sample_seed: int = Field(default=42, alias="SAMPLE_SEED")

    # Input files
    chunks_input_1k: str = Field(default="data/processed/corpus_1k_chunks.jsonl", alias="CHUNKS_INPUT_1K")
    chunks_input_10k: str = Field(default="data/processed/corpus_10k_chunks.jsonl", alias="CHUNKS_INPUT_10K")

    # LoRA adapter (for local LLM generation)
    local_lora_adapter_path: str | None = Field(default=None, alias="LOCAL_LORA_ADAPTER_PATH")
    

@lru_cache
def get_settings() -> Settings:
    return Settings()