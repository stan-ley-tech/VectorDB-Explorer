import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.index_cache import index_cache
from app.models import Collection, IndexConfig, VectorRecord
from app.schemas import CollectionCreate, CollectionOut, CollectionStats

router = APIRouter(prefix="/collections", tags=["collections"])


def get_collection_or_404(db: Session, collection_id: uuid.UUID) -> Collection:
    collection = db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="collection not found")
    return collection


@router.post("", response_model=CollectionOut, status_code=201)
def create_collection(payload: CollectionCreate, db: Session = Depends(get_db)) -> Collection:
    existing = db.scalar(select(Collection).where(Collection.name == payload.name))
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"collection '{payload.name}' already exists")

    collection = Collection(name=payload.name, dim=payload.dim, metric=payload.metric)
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return collection


@router.get("", response_model=list[CollectionOut])
def list_collections(db: Session = Depends(get_db)) -> list[Collection]:
    return list(db.scalars(select(Collection).order_by(Collection.created_at.desc())))


@router.get("/{collection_id}", response_model=CollectionOut)
def get_collection(collection_id: uuid.UUID, db: Session = Depends(get_db)) -> Collection:
    return get_collection_or_404(db, collection_id)


@router.get("/{collection_id}/stats", response_model=CollectionStats)
def get_collection_stats(collection_id: uuid.UUID, db: Session = Depends(get_db)) -> CollectionStats:
    collection = get_collection_or_404(db, collection_id)

    document_count = (
        db.scalar(select(func.count()).select_from(VectorRecord).where(VectorRecord.collection_id == collection_id))
        or 0
    )

    latest_index = db.scalar(
        select(IndexConfig)
        .where(IndexConfig.collection_id == collection_id, IndexConfig.status == "ready")
        .order_by(IndexConfig.built_at.desc())
    )

    # pg_total_relation_size would give an exact on-disk figure, but that's
    # a separate raw-SQL round trip per collection on every stats request.
    # This estimate (embedding floats + a small per-row overhead for the id,
    # foreign key and timestamp columns) is cheap and close enough for a
    # dashboard summary.
    storage_bytes = document_count * (collection.dim * 4 + 64)

    return CollectionStats(
        document_count=document_count,
        dimensions=collection.dim,
        metric=collection.metric,
        index_type=latest_index.index_type if latest_index else None,
        storage_bytes=storage_bytes,
    )


@router.delete("/{collection_id}", status_code=204)
def delete_collection(collection_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    collection = get_collection_or_404(db, collection_id)
    db.delete(collection)
    db.commit()
    index_cache.invalidate(collection_id)
