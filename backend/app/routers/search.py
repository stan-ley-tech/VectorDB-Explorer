import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.embeddings import HashingEmbedder
from app.engine_bridge import Metric, NativeEngineUnavailableError
from app.index_cache import index_cache
from app.models import IndexConfig, QueryLog, VectorMetadata, VectorRecord
from app.routers.collections import get_collection_or_404
from app.schemas import IndexCreate, IndexOut, SearchHitOut, SearchRequest, SearchResponse

router = APIRouter(prefix="/collections/{collection_id}", tags=["search"])


def _distance_expr(metric: Metric, query_vector: list[float]):
    column = VectorRecord.embedding
    if metric == Metric.COSINE:
        return column.cosine_distance(query_vector)
    if metric == Metric.EUCLIDEAN:
        return column.l2_distance(query_vector)
    if metric == Metric.DOT:
        return column.max_inner_product(query_vector)
    raise ValueError(f"unsupported metric: {metric}")


def _distance_to_score(metric: Metric, distance: float) -> float:
    """Converts a pgvector distance/operator result into the same "higher is
    better" score convention the native engine uses, so a result's score
    means the same thing whether use_index was true or false."""
    if metric == Metric.COSINE:
        return 1.0 - distance
    if metric == Metric.EUCLIDEAN:
        return -distance
    if metric == Metric.DOT:
        return -distance  # pgvector's <#> operator returns negative inner product
    raise ValueError(f"unsupported metric: {metric}")


def _pgvector_search(
    db: Session, collection_id: uuid.UUID, query_vector: list[float], top_k: int, metric: Metric, metadata_filter: dict[str, Any] | None
) -> list[SearchHitOut]:
    distance_col = _distance_expr(metric, query_vector).label("distance")

    stmt = (
        select(VectorRecord, VectorMetadata, distance_col)
        .outerjoin(VectorMetadata, VectorMetadata.vector_id == VectorRecord.id)
        .where(VectorRecord.collection_id == collection_id)
        .order_by(distance_col.asc())
        .limit(top_k)
    )
    if metadata_filter:
        stmt = stmt.where(VectorMetadata.data.contains(metadata_filter))

    results = []
    for record, meta, distance in db.execute(stmt).all():
        results.append(
            SearchHitOut(
                id=record.id,
                external_id=record.external_id,
                document=record.document,
                score=_distance_to_score(metric, float(distance)),
                metadata=meta.data if meta else {},
            )
        )
    return results


def _hydrate_indexed_hits(db: Session, ordered_ids: list[uuid.UUID], scores: list[float]) -> list[SearchHitOut]:
    if not ordered_ids:
        return []

    stmt = (
        select(VectorRecord, VectorMetadata)
        .outerjoin(VectorMetadata, VectorMetadata.vector_id == VectorRecord.id)
        .where(VectorRecord.id.in_(ordered_ids))
    )
    rows_by_id = {record.id: (record, meta) for record, meta in db.execute(stmt).all()}

    results = []
    for vector_id, score in zip(ordered_ids, scores):
        found = rows_by_id.get(vector_id)
        if found is None:
            continue  # deleted after the index snapshot was built
        record, meta = found
        results.append(
            SearchHitOut(
                id=record.id,
                external_id=record.external_id,
                document=record.document,
                score=score,
                metadata=meta.data if meta else {},
            )
        )
    return results


@router.post("/search", response_model=SearchResponse)
def search_collection(
    collection_id: uuid.UUID, payload: SearchRequest, db: Session = Depends(get_db)
) -> SearchResponse:
    collection = get_collection_or_404(db, collection_id)
    metric = Metric(payload.metric or collection.metric)

    if payload.query_vector is not None:
        if len(payload.query_vector) != collection.dim:
            raise HTTPException(
                status_code=422,
                detail=f"query_vector has dimension {len(payload.query_vector)}, expected {collection.dim}",
            )
        query_vector = payload.query_vector
    elif payload.query:
        query_vector = HashingEmbedder(dim=collection.dim).embed(payload.query)
    else:
        raise HTTPException(status_code=422, detail="either 'query' or 'query_vector' is required")

    used_index = False

    # The in-memory ANN index has no notion of metadata, so a filtered
    # search always falls back to the pgvector path even when an index is
    # cached and use_index was requested.
    cached = index_cache.get(collection_id) if (payload.use_index and payload.filter is None) else None

    start = time.perf_counter()
    if cached is not None:
        cached_ids, index_handle = cached
        hits = index_handle.search(query_vector, payload.top_k, payload.nprobe)
        ordered_ids = [cached_ids[hit.row] for hit in hits]
        scores = [hit.score for hit in hits]
        results = _hydrate_indexed_hits(db, ordered_ids, scores)
        used_index = True
    else:
        results = _pgvector_search(db, collection_id, query_vector, payload.top_k, metric, payload.filter)
    latency_ms = (time.perf_counter() - start) * 1000.0

    db.add(
        QueryLog(
            collection_id=collection_id,
            query_text=payload.query,
            metric=metric.value,
            top_k=payload.top_k,
            used_index=used_index,
            latency_ms=latency_ms,
            result_count=len(results),
        )
    )
    db.commit()

    return SearchResponse(results=results, latency_ms=latency_ms, used_index=used_index)


def _index_to_out(config: IndexConfig) -> IndexOut:
    return IndexOut(
        id=config.id,
        index_type=config.index_type,
        metric=config.metric,
        nlist=config.nlist,
        status=config.status,
        built_at=config.built_at,
        created_at=config.created_at,
    )


@router.post("/index", response_model=IndexOut, status_code=201)
def build_index(collection_id: uuid.UUID, payload: IndexCreate, db: Session = Depends(get_db)) -> IndexOut:
    collection = get_collection_or_404(db, collection_id)

    rows = db.execute(
        select(VectorRecord.id, VectorRecord.embedding).where(VectorRecord.collection_id == collection_id)
    ).all()
    if len(rows) < payload.nlist:
        raise HTTPException(
            status_code=422,
            detail=f"collection has {len(rows)} vectors, fewer than nlist={payload.nlist}; "
            "lower nlist or add more vectors first",
        )

    vector_ids = [row.id for row in rows]
    vectors = np.array([list(row.embedding) for row in rows], dtype=np.float32)

    index_config = IndexConfig(
        collection_id=collection_id,
        index_type=payload.index_type,
        metric=collection.metric,
        nlist=payload.nlist,
        status="building",
    )
    db.add(index_config)
    db.commit()
    db.refresh(index_config)

    try:
        index_cache.build(collection_id, vector_ids, vectors, Metric(collection.metric), payload.nlist)
    except NativeEngineUnavailableError as exc:
        index_config.status = "failed"
        db.commit()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    index_config.status = "ready"
    index_config.built_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(index_config)

    return _index_to_out(index_config)


@router.get("/index", response_model=IndexOut | None)
def get_index(collection_id: uuid.UUID, db: Session = Depends(get_db)) -> IndexOut | None:
    get_collection_or_404(db, collection_id)
    latest = db.scalar(
        select(IndexConfig)
        .where(IndexConfig.collection_id == collection_id)
        .order_by(IndexConfig.created_at.desc())
    )
    return _index_to_out(latest) if latest else None
