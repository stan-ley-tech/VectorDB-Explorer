"""Bridge to the native C++ search engine, with a pure-Python fallback.

The native module (vdb_native) is an optional compiled extension built from
engine/bindings (see engine/README or the top-level README for the build
command). It won't be present in every environment - a machine without a
C++ toolchain, for instance - so brute-force search degrades to a NumPy
implementation instead of failing outright. Approximate (IVF) search has no
such fallback: duplicating k-means clustering in two languages isn't worth
the maintenance cost for what's meant to be a benchmarking aid, so it raises
NativeEngineUnavailableError when the extension isn't built.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import vdb_native

    NATIVE_ENGINE_AVAILABLE = True
except ImportError:
    vdb_native = None
    NATIVE_ENGINE_AVAILABLE = False


class Metric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"


class NativeEngineUnavailableError(RuntimeError):
    pass


@dataclass
class SearchHit:
    """`row` is the 0-based index into the vectors array passed in, not a
    database id - callers map rows back to whatever id space they need."""

    row: int
    score: float


def _to_native_metric(metric: Metric):
    return {
        Metric.COSINE: vdb_native.Metric.COSINE,
        Metric.EUCLIDEAN: vdb_native.Metric.EUCLIDEAN,
        Metric.DOT: vdb_native.Metric.DOT_PRODUCT,
    }[metric]


def _numpy_scores(query: np.ndarray, vectors: np.ndarray, metric: Metric) -> np.ndarray:
    if metric == Metric.COSINE:
        query_unit = query / (np.linalg.norm(query) + 1e-12)
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return (vectors / vector_norms) @ query_unit
    if metric == Metric.DOT:
        return vectors @ query
    if metric == Metric.EUCLIDEAN:
        return -np.linalg.norm(vectors - query, axis=1)
    raise ValueError(f"unknown metric: {metric}")


def brute_force_search_numpy(vectors: np.ndarray, query: list[float], top_k: int, metric: Metric) -> list[SearchHit]:
    if vectors.shape[0] == 0 or top_k <= 0:
        return []

    query_arr = np.asarray(query, dtype=np.float32)
    scores = _numpy_scores(query_arr, vectors, metric)

    k = min(top_k, vectors.shape[0])
    candidate_indices = np.argpartition(-scores, k - 1)[:k]
    ordered = candidate_indices[np.argsort(-scores[candidate_indices])]
    return [SearchHit(row=int(i), score=float(scores[i])) for i in ordered]


def brute_force_search(vectors: np.ndarray, query: list[float], top_k: int, metric: Metric) -> list[SearchHit]:
    if not NATIVE_ENGINE_AVAILABLE:
        return brute_force_search_numpy(vectors, query, top_k, metric)

    store = vdb_native.VectorStore(vectors.shape[1])
    for i, row in enumerate(vectors):
        store.upsert(i, row.tolist())
    results = vdb_native.brute_force_search(store, list(query), top_k, _to_native_metric(metric))
    return [SearchHit(row=r.id, score=r.score) for r in results]


def estimate_memory_bytes(vectors: np.ndarray) -> int:
    return int(vectors.nbytes)


class IVFIndexHandle:
    """Owns both the native VectorStore and IVFIndex built over it - the
    engine's search() call needs the store alongside the index, so the two
    travel together here instead of leaking that detail to callers."""

    def __init__(self, store: "vdb_native.VectorStore", index: "vdb_native.IVFIndex"):
        self._store = store
        self._index = index

    def search(self, query: list[float], top_k: int, nprobe: int) -> list[SearchHit]:
        results = self._index.search(self._store, list(query), top_k, nprobe)
        return [SearchHit(row=r.id, score=r.score) for r in results]

    def cluster_sizes(self) -> list[int]:
        return self._index.cluster_sizes()

    def memory_usage_bytes(self) -> int:
        return self._store.memory_usage_bytes()


def build_ivf_index(
    vectors: np.ndarray, metric: Metric, nlist: int, kmeans_iterations: int = 15, seed: int = 42
) -> IVFIndexHandle:
    if not NATIVE_ENGINE_AVAILABLE:
        raise NativeEngineUnavailableError(
            "approximate (IVF) search requires the compiled vdb_native extension; "
            "build engine/ with -DVDB_BUILD_PYTHON_BINDINGS=ON and install it into this environment"
        )

    dim = vectors.shape[1]
    store = vdb_native.VectorStore(dim)
    for i, row in enumerate(vectors):
        store.upsert(i, row.tolist())

    index = vdb_native.IVFIndex(dim, _to_native_metric(metric), nlist)
    index.build(store, kmeans_iterations, seed)
    return IVFIndexHandle(store, index)
