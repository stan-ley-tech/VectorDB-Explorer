import csv
import io
import json
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.embeddings import HashingEmbedder
from app.index_cache import index_cache
from app.ingestion import DatasetFormatError, DatasetRecord, batched, parse_csv, parse_json
from app.models import Collection, VectorMetadata, VectorRecord
from app.routers.collections import get_collection_or_404
from app.schemas import ImportResponse, VectorBulkIn, VectorIn, VectorOut, VectorUpdate

router = APIRouter(prefix="/collections/{collection_id}", tags=["vectors"])


def _to_vector_out(record: VectorRecord) -> VectorOut:
    return VectorOut(
        id=record.id,
        external_id=record.external_id,
        document=record.document,
        metadata=record.vector_metadata.data if record.vector_metadata else {},
        created_at=record.created_at,
    )


def _resolve_embedding(item: VectorIn, dim: int, embedder: HashingEmbedder) -> list[float]:
    if item.embedding is not None:
        if len(item.embedding) != dim:
            raise HTTPException(
                status_code=422,
                detail=f"embedding for '{item.external_id}' has dimension {len(item.embedding)}, expected {dim}",
            )
        return item.embedding
    if not item.document:
        raise HTTPException(
            status_code=422,
            detail=f"vector '{item.external_id}' has neither an embedding nor a document to embed",
        )
    return embedder.embed(item.document)


def _find_existing(db: Session, collection_id: uuid.UUID, external_id: str) -> VectorRecord | None:
    return db.scalar(
        select(VectorRecord).where(
            VectorRecord.collection_id == collection_id, VectorRecord.external_id == external_id
        )
    )


def _apply_upsert(
    db: Session, collection_id: uuid.UUID, external_id: str, document: str | None, embedding: list[float], metadata: dict
) -> VectorRecord:
    existing = _find_existing(db, collection_id, external_id)
    if existing is not None:
        existing.document = document
        existing.embedding = embedding
        if existing.vector_metadata is None:
            existing.vector_metadata = VectorMetadata(data=metadata)
        else:
            existing.vector_metadata.data = metadata
        return existing

    record = VectorRecord(collection_id=collection_id, external_id=external_id, document=document, embedding=embedding)
    record.vector_metadata = VectorMetadata(data=metadata)
    db.add(record)
    return record


@router.post("/vectors", response_model=list[VectorOut], status_code=201)
def upsert_vectors(
    collection_id: uuid.UUID, payload: VectorBulkIn, db: Session = Depends(get_db)
) -> list[VectorOut]:
    collection = get_collection_or_404(db, collection_id)
    embedder = HashingEmbedder(dim=collection.dim)

    records = [
        _apply_upsert(
            db,
            collection_id,
            item.external_id,
            item.document,
            _resolve_embedding(item, collection.dim, embedder),
            item.metadata,
        )
        for item in payload.vectors
    ]

    db.commit()
    for record in records:
        db.refresh(record)
    index_cache.invalidate(collection_id)

    return [_to_vector_out(r) for r in records]


@router.get("/vectors/{vector_id}", response_model=VectorOut)
def get_vector(collection_id: uuid.UUID, vector_id: uuid.UUID, db: Session = Depends(get_db)) -> VectorOut:
    get_collection_or_404(db, collection_id)
    record = db.get(VectorRecord, vector_id)
    if record is None or record.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="vector not found")
    return _to_vector_out(record)


@router.patch("/vectors/{vector_id}", response_model=VectorOut)
def update_vector(
    collection_id: uuid.UUID, vector_id: uuid.UUID, payload: VectorUpdate, db: Session = Depends(get_db)
) -> VectorOut:
    collection = get_collection_or_404(db, collection_id)
    record = db.get(VectorRecord, vector_id)
    if record is None or record.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="vector not found")

    if payload.document is not None:
        record.document = payload.document
    if payload.embedding is not None:
        if len(payload.embedding) != collection.dim:
            raise HTTPException(
                status_code=422,
                detail=f"embedding has dimension {len(payload.embedding)}, expected {collection.dim}",
            )
        record.embedding = payload.embedding
    if payload.metadata is not None:
        if record.vector_metadata is None:
            record.vector_metadata = VectorMetadata(data=payload.metadata)
        else:
            record.vector_metadata.data = payload.metadata

    db.commit()
    db.refresh(record)
    index_cache.invalidate(collection_id)
    return _to_vector_out(record)


@router.delete("/vectors/{vector_id}", status_code=204)
def delete_vector(collection_id: uuid.UUID, vector_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    get_collection_or_404(db, collection_id)
    record = db.get(VectorRecord, vector_id)
    if record is None or record.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="vector not found")
    db.delete(record)
    db.commit()
    index_cache.invalidate(collection_id)


@router.post("/import", response_model=ImportResponse)
async def import_dataset(
    collection_id: uuid.UUID,
    file: UploadFile,
    format: Literal["csv", "json"] = Query(...),
    text_field: str = Query("text"),
    id_field: str | None = Query(None),
    db: Session = Depends(get_db),
) -> ImportResponse:
    collection = get_collection_or_404(db, collection_id)
    embedder = HashingEmbedder(dim=collection.dim)

    raw = await file.read()
    text_stream = io.StringIO(raw.decode("utf-8"))

    try:
        record_iter = (
            parse_csv(text_stream, text_field=text_field, id_field=id_field)
            if format == "csv"
            else parse_json(text_stream, text_field=text_field, id_field=id_field)
        )
    except DatasetFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    imported = 0
    failed = 0
    errors: list[str] = []

    # Commit once per batch rather than once per row (too many round trips)
    # or once for the whole file (one bad row would roll back everything
    # already imported). A batch-level commit also means a single failed
    # row inside a batch doesn't affect rows in other batches.
    for batch in batched(record_iter, settings.ingest_batch_size):
        for item in batch:
            try:
                _import_one(db, collection_id, item, embedder)
                imported += 1
            except Exception as exc:
                failed += 1
                if len(errors) < 20:
                    errors.append(f"{item.external_id}: {exc}")
        db.commit()

    index_cache.invalidate(collection_id)
    return ImportResponse(imported=imported, failed=failed, errors=errors)


def _import_one(db: Session, collection_id: uuid.UUID, item: DatasetRecord, embedder: HashingEmbedder) -> None:
    if not item.document:
        raise DatasetFormatError(f"record '{item.external_id}' has no text to embed")
    embedding = embedder.embed(item.document)
    _apply_upsert(db, collection_id, item.external_id, item.document, embedding, item.metadata)


@router.get("/export")
def export_dataset(
    collection_id: uuid.UUID,
    format: Literal["csv", "json"] = Query("json"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    get_collection_or_404(db, collection_id)

    records = db.scalars(
        select(VectorRecord).where(VectorRecord.collection_id == collection_id).order_by(VectorRecord.created_at)
    )

    if format == "json":
        # Includes the full embedding for a true round trip (re-import with
        # embeddings supplied directly, skipping re-generation).
        def generate_json():
            yield "["
            first = True
            for record in records:
                if not first:
                    yield ","
                first = False
                yield json.dumps(
                    {
                        "external_id": record.external_id,
                        "document": record.document,
                        "embedding": list(record.embedding),
                        "metadata": record.vector_metadata.data if record.vector_metadata else {},
                    }
                )
            yield "]"

        return StreamingResponse(generate_json(), media_type="application/json")

    # CSV export deliberately omits the embedding column: CSV isn't a good
    # fit for a few hundred float columns, and re-importing from CSV will
    # just regenerate the embedding from the document text anyway.
    def generate_csv():
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["external_id", "text", "metadata"])
        yield buffer.getvalue()
        for record in records:
            buffer.seek(0)
            buffer.truncate(0)
            metadata = record.vector_metadata.data if record.vector_metadata else {}
            writer.writerow([record.external_id, record.document or "", json.dumps(metadata)])
            yield buffer.getvalue()

    return StreamingResponse(generate_csv(), media_type="text/csv")
