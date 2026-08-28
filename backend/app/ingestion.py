import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import IO, Any


class DatasetFormatError(ValueError):
    """Raised when a dataset file doesn't match the expected shape."""


@dataclass
class DatasetRecord:
    external_id: str
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_csv(stream: IO[str], text_field: str = "text", id_field: str | None = None) -> Iterator[DatasetRecord]:
    """Streams a CSV file row by row via csv.DictReader, so import memory use
    stays bounded by one row (plus whatever batch size the caller chooses)
    instead of the whole file."""
    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
        raise DatasetFormatError("CSV file has no header row")
    if text_field not in reader.fieldnames:
        raise DatasetFormatError(f"CSV file is missing required column '{text_field}'")

    for i, row in enumerate(reader):
        external_id = row.get(id_field) if id_field else None
        if not external_id:
            external_id = str(i)
        document = row.get(text_field) or ""
        metadata = {k: v for k, v in row.items() if k not in (text_field, id_field) and v is not None}
        yield DatasetRecord(external_id=str(external_id), document=document, metadata=metadata)


def parse_json(stream: IO[str], text_field: str = "text", id_field: str | None = None) -> Iterator[DatasetRecord]:
    """Parses a JSON array (or an object with a top-level "records"/"data"
    array) of documents. Unlike parse_csv this reads the whole file into
    memory up front, since json.load has no incremental mode without an
    extra streaming-JSON dependency - fine for the moderate dataset sizes
    this tool targets, but CSV is the better choice for very large imports.
    """
    data = json.load(stream)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("records", data.get("data"))
        if records is None:
            raise DatasetFormatError("JSON object must contain a 'records' or 'data' array")
    else:
        raise DatasetFormatError("JSON dataset must be an array or an object with a 'records' array")

    for i, row in enumerate(records):
        if not isinstance(row, dict):
            raise DatasetFormatError(f"record at index {i} is not an object")
        if text_field not in row:
            raise DatasetFormatError(f"record at index {i} is missing required field '{text_field}'")

        external_id = row.get(id_field) if id_field else None
        if not external_id:
            external_id = str(i)
        document = row.get(text_field) or ""
        metadata = {k: v for k, v in row.items() if k not in (text_field, id_field)}
        yield DatasetRecord(external_id=str(external_id), document=str(document), metadata=metadata)


def batched(records: Iterator[DatasetRecord], batch_size: int) -> Iterator[list[DatasetRecord]]:
    """Groups a record stream into fixed-size batches so a caller (e.g. the
    bulk insert endpoint) can commit to the database in bounded chunks
    instead of holding an entire dataset in one transaction."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batch: list[DatasetRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
