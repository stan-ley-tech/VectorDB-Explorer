"""initial schema

Revision ID: e1e39c48e7bb
Revises:
Create Date: 2026-02-03 00:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "e1e39c48e7bb"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "collections",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("dim", sa.Integer, nullable=False),
        sa.Column("metric", sa.String(32), nullable=False, server_default="cosine"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "vectors",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "collection_id",
            UUID(as_uuid=True),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("external_id", sa.String(255), nullable=False),
        sa.Column("document", sa.Text, nullable=True),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_vectors_collection_id", "vectors", ["collection_id"])
    op.create_index(
        "ix_vectors_collection_external_id",
        "vectors",
        ["collection_id", "external_id"],
        unique=True,
    )

    op.create_table(
        "vector_metadata",
        sa.Column(
            "vector_id",
            UUID(as_uuid=True),
            sa.ForeignKey("vectors.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("data", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index(
        "ix_vector_metadata_data", "vector_metadata", ["data"], postgresql_using="gin"
    )

    op.create_table(
        "indexes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "collection_id",
            UUID(as_uuid=True),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("index_type", sa.String(32), nullable=False, server_default="ivf"),
        sa.Column("metric", sa.String(32), nullable=False),
        sa.Column("nlist", sa.Integer, nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("built_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "queries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "collection_id",
            UUID(as_uuid=True),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query_text", sa.Text, nullable=True),
        sa.Column("metric", sa.String(32), nullable=False),
        sa.Column("top_k", sa.Integer, nullable=False),
        sa.Column("used_index", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("result_count", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_queries_collection_created_at", "queries", ["collection_id", "created_at"]
    )

    op.create_table(
        "benchmarks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "collection_id",
            UUID(as_uuid=True),
            sa.ForeignKey("collections.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("config", sa.JSON, nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )

    op.create_table(
        "benchmark_results",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "benchmark_id",
            UUID(as_uuid=True),
            sa.ForeignKey("benchmarks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("avg_latency_ms", sa.Float, nullable=False),
        sa.Column("p95_latency_ms", sa.Float, nullable=False),
        sa.Column("p99_latency_ms", sa.Float, nullable=False),
        sa.Column("throughput_qps", sa.Float, nullable=False),
        sa.Column("recall_at_k", sa.Float, nullable=True),
        sa.Column("memory_bytes", sa.BigInteger, nullable=False),
        sa.Column("index_build_ms", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("benchmark_results")
    op.drop_table("benchmarks")
    op.drop_index("ix_queries_collection_created_at", table_name="queries")
    op.drop_table("queries")
    op.drop_table("indexes")
    op.drop_index("ix_vector_metadata_data", table_name="vector_metadata")
    op.drop_table("vector_metadata")
    op.drop_index("ix_vectors_collection_external_id", table_name="vectors")
    op.drop_index("ix_vectors_collection_id", table_name="vectors")
    op.drop_table("vectors")
    op.drop_table("collections")
