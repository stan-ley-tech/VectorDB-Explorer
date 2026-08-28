import uuid

import pytest

from app.engine_bridge import NativeEngineUnavailableError
from app.index_cache import IndexCache


def test_get_on_empty_cache_returns_none():
    cache = IndexCache()
    assert cache.get(uuid.uuid4()) is None


def test_invalidate_on_missing_entry_is_a_no_op():
    cache = IndexCache()
    cache.invalidate(uuid.uuid4())  # must not raise


def test_build_without_native_engine_raises_and_does_not_populate_cache():
    import numpy as np

    from app.engine_bridge import Metric

    cache = IndexCache()
    collection_id = uuid.uuid4()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    with pytest.raises(NativeEngineUnavailableError):
        cache.build(collection_id, [uuid.uuid4()] * 3, vectors, Metric.COSINE, nlist=2)

    assert cache.get(collection_id) is None
