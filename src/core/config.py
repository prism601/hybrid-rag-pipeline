"""
Configuration management for the RAG pipeline
"""
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with validation"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # LLM Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    llm_model: str = Field(default="gpt-4-turbo-preview", description="LLM model to use")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2000, ge=1)

    # Vector Database
    vector_db_type: Literal["chromadb", "pinecone", "qdrant"] = Field(default="chromadb")
    pinecone_api_key: str = Field(default="")
    pinecone_environment: str = Field(default="us-west1-gcp")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")

    # Knowledge Graph
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    # Redis Cache
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: str = Field(default="")
    cache_ttl: int = Field(default=3600)

    # Elasticsearch
    elasticsearch_url: str = Field(default="http://localhost:9200")
    elasticsearch_user: str = Field(default="")
    elasticsearch_password: str = Field(default="")

    # Embedding Models
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Chunking Configuration
    default_chunk_size: int = Field(default=512, ge=100, le=2048)
    default_chunk_overlap: int = Field(default=50, ge=0)
    auto_optimize_chunks: bool = Field(default=True)

    # Retrieval Configuration
    top_k_retrieval: int = Field(default=10, ge=1)
    rerank_top_k: int = Field(default=5, ge=1)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    use_mmr: bool = Field(default=True)
    mmr_diversity_score: float = Field(default=0.3, ge=0.0, le=1.0)

    # Hallucination Detection
    hallucination_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_faithfulness_check: bool = Field(default=True)

    # Evaluation
    enable_ragas_metrics: bool = Field(default=True)
    evaluation_batch_size: int = Field(default=10)

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    enable_cors: bool = Field(default=True)
    api_rate_limit: int = Field(default=100)

    # Monitoring
    enable_prometheus: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    log_level: str = Field(default="INFO")
    sentry_dsn: str = Field(default="")

    # Performance
    max_concurrent_requests: int = Field(default=50)
    batch_size: int = Field(default=32)
    enable_async_processing: bool = Field(default=True)

    # A/B Testing
    enable_ab_testing: bool = Field(default=True)
    ab_test_traffic_split: float = Field(default=0.5, ge=0.0, le=1.0)


# Global settings instance
settings = Settings()
