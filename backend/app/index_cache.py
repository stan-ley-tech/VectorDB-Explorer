"""In-memory cache of built ANN indexes, keyed by collection id.

pgvector holds the durable copy of every vector; this cache holds a
snapshot loaded into the native engine's VectorStore + IVFIndex so a
`use_index=true` search doesn't rebuild the index on every request. It's
process-local and unbounded by design - a handful of collections, each up
to a few hundred thousand vectors, comfortably fits in memory for a
single-instance deployment. A multi-instance deployment would need a shared
cache or a rebuild per instance; that's out of scope here.

An entry is invalidated whenever its collection's vectors change (add,
update, delete) so a stale index is never silently served - the caller has
to rebuild via POST /collections/{id}/index before use_index=true works
again, the same "rebuild after mutation" contract IVFIndex itself enforces
with its store version check.
"""

import threading
import uuid

import numpy as np

from app.engine_bridge import IVFIndexHandle, Metric, build_ivf_index


class IndexCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[uuid.UUID, tuple[list[uuid.UUID], IVFIndexHandle]] = {}

    def build(
        self,
        collection_id: uuid.UUID,
        vector_ids: list[uuid.UUID],
        vectors: np.ndarray,
        metric: Metric,
        nlist: int,
    ) -> IVFIndexHandle:
        handle = build_ivf_index(vectors, metric, nlist)
        with self._lock:
            self._entries[collection_id] = (vector_ids, handle)
        return handle

    def get(self, collection_id: uuid.UUID) -> tuple[list[uuid.UUID], IVFIndexHandle] | None:
        with self._lock:
            return self._entries.get(collection_id)

    def invalidate(self, collection_id: uuid.UUID) -> None:
        with self._lock:
            self._entries.pop(collection_id, None)


# Process-wide singleton: every request handler shares one cache so an
# index built by one request is reusable by the next.
index_cache = IndexCache()
