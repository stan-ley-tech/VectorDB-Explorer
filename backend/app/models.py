import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    dim: Mapped[int] = mapped_column(nullable=False)
    metric: Mapped[str] = mapped_column(String(32), nullable=False, default="cosine")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    vectors: Mapped[list["VectorRecord"]] = relationship(back_populates="collection", cascade="all, delete-orphan")
    indexes: Mapped[list["IndexConfig"]] = relationship(back_populates="collection", cascade="all, delete-orphan")


class VectorRecord(Base):
    __tablename__ = "vectors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    document: Mapped[str | None] = mapped_column(nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    collection: Mapped[Collection] = relationship(back_populates="vectors")
    vector_metadata: Mapped["VectorMetadata"] = relationship(
        back_populates="vector", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (
        Index("ix_vectors_collection_id", "collection_id"),
        Index("ix_vectors_collection_external_id", "collection_id", "external_id", unique=True),
    )


class VectorMetadata(Base):
    """One JSONB payload per vector, kept in its own table rather than a
    column on `vectors` so a GIN index over `data` doesn't have to be
    maintained alongside every embedding write."""

    __tablename__ = "vector_metadata"

    vector_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vectors.id", ondelete="CASCADE"), primary_key=True
    )
    data: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    vector: Mapped[VectorRecord] = relationship(back_populates="vector_metadata")

    __table_args__ = (Index("ix_vector_metadata_data", "data", postgresql_using="gin"),)


class IndexConfig(Base):
    __tablename__ = "indexes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    index_type: Mapped[str] = mapped_column(String(32), nullable=False, default="ivf")
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    nlist: Mapped[int | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    built_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    collection: Mapped[Collection] = relationship(back_populates="indexes")


class QueryLog(Base):
    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    query_text: Mapped[str | None] = mapped_column(nullable=True)
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    top_k: Mapped[int] = mapped_column(nullable=False)
    used_index: Mapped[bool] = mapped_column(default=False)
    latency_ms: Mapped[float] = mapped_column(nullable=False)
    result_count: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (Index("ix_queries_collection_created_at", "collection_id", "created_at"),)


class Benchmark(Base):
    __tablename__ = "benchmarks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("collections.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    results: Mapped[list["BenchmarkResult"]] = relationship(back_populates="benchmark", cascade="all, delete-orphan")


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    benchmark_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("benchmarks.id", ondelete="CASCADE"), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)  # "brute_force" | "indexed"
    avg_latency_ms: Mapped[float] = mapped_column(nullable=False)
    p95_latency_ms: Mapped[float] = mapped_column(nullable=False)
    p99_latency_ms: Mapped[float] = mapped_column(nullable=False)
    throughput_qps: Mapped[float] = mapped_column(nullable=False)
    recall_at_k: Mapped[float | None] = mapped_column(nullable=True)
    memory_bytes: Mapped[int] = mapped_column(nullable=False)
    index_build_ms: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    benchmark: Mapped[Benchmark] = relationship(back_populates="results")
