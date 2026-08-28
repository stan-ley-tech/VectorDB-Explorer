import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

MetricName = Literal["cosine", "euclidean", "dot"]


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    dim: int = Field(gt=0)
    metric: MetricName = "cosine"


class CollectionOut(BaseModel):
    id: uuid.UUID
    name: str
    dim: int
    metric: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CollectionStats(BaseModel):
    document_count: int
    dimensions: int
    metric: str
    index_type: str | None
    storage_bytes: int


class VectorIn(BaseModel):
    external_id: str = Field(min_length=1, max_length=255)
    document: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorBulkIn(BaseModel):
    vectors: list[VectorIn] = Field(min_length=1, max_length=5000)


class VectorOut(BaseModel):
    id: uuid.UUID
    external_id: str
    document: str | None
    metadata: dict[str, Any]
    created_at: datetime


class VectorUpdate(BaseModel):
    document: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None


class ImportResponse(BaseModel):
    imported: int
    failed: int
    errors: list[str]


class SearchRequest(BaseModel):
    query: str | None = None
    query_vector: list[float] | None = None
    top_k: int = Field(default=10, gt=0, le=1000)
    metric: MetricName | None = None
    filter: dict[str, Any] | None = None
    use_index: bool = False
    nprobe: int = Field(default=4, gt=0)


class SearchHitOut(BaseModel):
    id: uuid.UUID
    external_id: str
    document: str | None
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchHitOut]
    latency_ms: float
    used_index: bool


class IndexCreate(BaseModel):
    index_type: Literal["ivf"] = "ivf"
    nlist: int = Field(default=100, gt=0)


class IndexOut(BaseModel):
    id: uuid.UUID
    index_type: str
    metric: str
    nlist: int | None
    status: str
    built_at: datetime | None
    created_at: datetime


class BenchmarkConfigIn(BaseModel):
    dataset_size: int = Field(gt=0, le=1_000_000)
    dimensions: int = Field(gt=0, le=4096)
    top_k: int = Field(default=10, gt=0)
    metric: MetricName = "cosine"
    nlist: int = Field(default=100, gt=0)
    nprobe: int = Field(default=8, gt=0)
    num_queries: int = Field(default=200, gt=0, le=20_000)
    concurrency: int = Field(default=4, gt=0, le=64)


class BenchmarkCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    config: BenchmarkConfigIn


class BenchmarkResultOut(BaseModel):
    mode: Literal["brute_force", "indexed"]
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    recall_at_k: float | None
    memory_bytes: int
    index_build_ms: float | None


class BenchmarkOut(BaseModel):
    id: uuid.UUID
    name: str
    config: dict[str, Any]
    status: str
    created_at: datetime
    completed_at: datetime | None
    results: list[BenchmarkResultOut]
