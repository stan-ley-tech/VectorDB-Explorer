import uuid

import numpy as np
import pytest

import app.engine_bridge as engine_bridge
from app.engine_bridge import Metric, NativeEngineUnavailableError
from app.index_cache import IndexCache
from tests.conftest import requires_native_engine


def test_get_on_empty_cache_returns_none():
    cache = IndexCache()
    assert cache.get(uuid.uuid4()) is None


def test_invalidate_on_missing_entry_is_a_no_op():
    cache = IndexCache()
    cache.invalidate(uuid.uuid4())  # must not raise


def test_build_without_native_engine_raises_and_does_not_populate_cache(monkeypatch):
    monkeypatch.setattr(engine_bridge, "NATIVE_ENGINE_AVAILABLE", False)

    cache = IndexCache()
    collection_id = uuid.uuid4()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    with pytest.raises(NativeEngineUnavailableError):
        cache.build(collection_id, [uuid.uuid4()] * 3, vectors, Metric.COSINE, nlist=2)

    assert cache.get(collection_id) is None


@requires_native_engine
def test_build_and_get_round_trips_through_the_cache():
    cache = IndexCache()
    collection_id = uuid.uuid4()
    vector_ids = [uuid.uuid4() for _ in range(4)]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.9, 0.1]], dtype=np.float32)

    cache.build(collection_id, vector_ids, vectors, Metric.COSINE, nlist=2)

    entry = cache.get(collection_id)
    assert entry is not None
    cached_ids, handle = entry
    assert cached_ids == vector_ids

    hits = handle.search([1.0, 0.0], top_k=1, nprobe=2)
    assert vector_ids[hits[0].row] == vector_ids[0]

    cache.invalidate(collection_id)
    assert cache.get(collection_id) is None
