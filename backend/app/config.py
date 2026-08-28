import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    db_pool_size: int
    db_max_overflow: int
    default_embedding_dim: int
    ingest_batch_size: int


def load_settings() -> Settings:
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg://vectordb:vectordb@localhost:5432/vectordb",
        ),
        db_pool_size=int(os.environ.get("DB_POOL_SIZE", "10")),
        db_max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "20")),
        default_embedding_dim=int(os.environ.get("DEFAULT_EMBEDDING_DIM", "256")),
        ingest_batch_size=int(os.environ.get("INGEST_BATCH_SIZE", "500")),
    )


settings = load_settings()
