import io

import pytest

from app.ingestion import DatasetFormatError, batched, parse_csv, parse_json


def test_parse_csv_reads_text_and_metadata_columns():
    csv_text = "text,category,author\n" "How to auth,security,alice\n" "Deploying to prod,ops,bob\n"

    records = list(parse_csv(io.StringIO(csv_text)))

    assert len(records) == 2
    assert records[0].document == "How to auth"
    assert records[0].metadata == {"category": "security", "author": "alice"}
    assert records[0].external_id == "0"
    assert records[1].external_id == "1"


def test_parse_csv_uses_id_field_when_given():
    csv_text = "id,text\n" "doc-1,hello\n" "doc-2,world\n"

    records = list(parse_csv(io.StringIO(csv_text), id_field="id"))

    assert [r.external_id for r in records] == ["doc-1", "doc-2"]
    assert records[0].metadata == {}


def test_parse_csv_rejects_missing_text_column():
    csv_text = "title,body\nsomething,else\n"
    with pytest.raises(DatasetFormatError):
        list(parse_csv(io.StringIO(csv_text)))


def test_parse_csv_rejects_empty_file():
    with pytest.raises(DatasetFormatError):
        list(parse_csv(io.StringIO("")))


def test_parse_json_accepts_a_bare_array():
    payload = '[{"text": "first"}, {"text": "second", "tag": "x"}]'

    records = list(parse_json(io.StringIO(payload)))

    assert len(records) == 2
    assert records[1].document == "second"
    assert records[1].metadata == {"tag": "x"}


def test_parse_json_accepts_an_object_with_records_key():
    payload = '{"records": [{"text": "only one"}]}'

    records = list(parse_json(io.StringIO(payload)))

    assert len(records) == 1
    assert records[0].document == "only one"


def test_parse_json_rejects_records_missing_text_field():
    payload = '[{"body": "no text field here"}]'
    with pytest.raises(DatasetFormatError):
        list(parse_json(io.StringIO(payload)))


def test_parse_json_rejects_non_object_records():
    payload = '["just a string"]'
    with pytest.raises(DatasetFormatError):
        list(parse_json(io.StringIO(payload)))


def test_parse_json_rejects_top_level_scalar():
    with pytest.raises(DatasetFormatError):
        list(parse_json(io.StringIO("42")))


def test_batched_groups_records_by_batch_size():
    csv_text = "text\n" + "\n".join(f"doc {i}" for i in range(5))
    records = parse_csv(io.StringIO(csv_text))

    batches = list(batched(records, batch_size=2))

    assert [len(b) for b in batches] == [2, 2, 1]


def test_batched_rejects_non_positive_batch_size():
    with pytest.raises(ValueError):
        list(batched(iter([]), batch_size=0))
