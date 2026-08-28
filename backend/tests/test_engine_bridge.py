import numpy as np
import pytest

from app.engine_bridge import (
    Metric,
    NativeEngineUnavailableError,
    brute_force_search,
    brute_force_search_numpy,
    build_ivf_index,
    estimate_memory_bytes,
)


def fixture_vectors() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],  # row 0: identical to the query
            [0.0, 1.0],  # row 1: orthogonal
            [-1.0, 0.0],  # row 2: opposite
            [0.9, 0.1],  # row 3: near-identical
        ],
        dtype=np.float32,
    )


def test_numpy_brute_force_ranks_by_cosine_similarity():
    hits = brute_force_search_numpy(fixture_vectors(), [1.0, 0.0], top_k=2, metric=Metric.COSINE)
    assert [h.row for h in hits] == [0, 3]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_numpy_brute_force_ranks_by_euclidean_distance():
    hits = brute_force_search_numpy(fixture_vectors(), [1.0, 0.0], top_k=2, metric=Metric.EUCLIDEAN)
    assert [h.row for h in hits] == [0, 3]


def test_numpy_brute_force_ranks_by_dot_product():
    hits = brute_force_search_numpy(fixture_vectors(), [1.0, 0.0], top_k=2, metric=Metric.DOT)
    assert [h.row for h in hits] == [0, 3]


def test_numpy_brute_force_on_empty_vectors_returns_no_hits():
    empty = np.empty((0, 4), dtype=np.float32)
    assert brute_force_search_numpy(empty, [1.0, 0.0, 0.0, 0.0], top_k=5, metric=Metric.COSINE) == []


def test_numpy_brute_force_top_k_zero_returns_no_hits():
    assert brute_force_search_numpy(fixture_vectors(), [1.0, 0.0], top_k=0, metric=Metric.COSINE) == []


def test_brute_force_search_dispatches_to_numpy_when_native_missing():
    # This test environment has no compiled vdb_native extension, so
    # brute_force_search must fall back to the numpy path and agree with it.
    hits_a = brute_force_search(fixture_vectors(), [1.0, 0.0], top_k=2, metric=Metric.COSINE)
    hits_b = brute_force_search_numpy(fixture_vectors(), [1.0, 0.0], top_k=2, metric=Metric.COSINE)
    assert hits_a == hits_b


def test_build_ivf_index_without_native_engine_raises_clear_error():
    with pytest.raises(NativeEngineUnavailableError):
        build_ivf_index(fixture_vectors(), Metric.COSINE, nlist=2)


def test_estimate_memory_bytes_matches_numpy_nbytes():
    vectors = fixture_vectors()
    assert estimate_memory_bytes(vectors) == vectors.nbytes
