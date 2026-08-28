# Architecture

## Components

```
┌─────────────┐      HTTP        ┌──────────────┐     SQL      ┌─────────────────┐
│   frontend   │ ───────────────▶│   backend    │──────────────▶│ Postgres+pgvector│
│ React + Vite │                  │   FastAPI    │               └─────────────────┘
└─────────────┘                  │              │
                                  │  ┌────────┐  │   pybind11
                                  │  │ engine  │◀─┼──────────────  engine/ (C++)
                                  │  │ bridge  │  │
                                  │  └────────┘  │
                                  └──────────────┘
```

- **frontend** (`frontend/`) - a small React/TypeScript dashboard: collections
  list, per-collection search, and benchmark configuration/results. Talks to
  the backend over plain `fetch`, nothing fancier.
- **backend** (`backend/`) - a FastAPI service that owns the API, the
  Postgres schema (via SQLAlchemy + Alembic), embedding generation, dataset
  import/export, and benchmark orchestration.
- **engine** (`engine/`) - a standalone C++17 vector search library
  (distance metrics, brute-force top-K, an IVF approximate index), exposed
  to Python through a pybind11 module. It doesn't know anything about
  Postgres, HTTP, or the rest of the system - it operates on plain
  `float`/`uint64_t` arrays.

## Why two search paths

The brief for this project was explicit about not relying entirely on
pgvector, so there are two independent ways a query gets answered:

1. **pgvector, exact.** `POST /collections/{id}/search` with
   `use_index=false` (the default) runs a real SQL query: `ORDER BY
   embedding <=> query LIMIT k`, filtered by collection and optionally by a
   JSONB metadata containment check. This is the correctness baseline and
   what the API uses unless you ask for the index.
2. **The C++ engine, approximate.** `POST /collections/{id}/index` builds an
   IVF index over a collection's current vectors using the native engine
   (k-means clustering into `nlist` partitions), and caches it in-process
   (`app/index_cache.py`). A search with `use_index=true` probes the
   `nprobe` nearest partitions instead of scanning everything.

The benchmark endpoint (`POST /collections/{id}/benchmark`) runs both
strategies back-to-back over the same synthetic dataset and query set, so
the latency/recall/memory numbers it reports are a direct, apples-to-apples
comparison - not two different code paths measured under different
conditions.

## Why vectors aren't dimension-constrained in Postgres

pgvector's `vector(n)` typmod (and the ivfflat/hnsw index types that require
it) assumes every row in a table has the same dimension. This project
supports collections of different dimensions living in one shared `vectors`
table, so the embedding column is declared as an unconstrained `vector`
instead - correct for any dimension, at the cost of not being able to build
a native pgvector ANN index on it. That's fine here: the ANN half of the
system is the C++ engine, not pgvector.

## Why a hashing-trick embedder instead of a hosted model

`app/embeddings.py` implements a deterministic, dependency-free embedder
using the hashing trick (tokens and bigrams hashed into fixed-size buckets,
signed to reduce collision bias, L2-normalized). It's not a substitute for a
trained model's semantics, but it means the whole system - storage,
indexing, search, benchmarking, tests, CI - runs offline with no API keys,
no GPU, and no network calls. Swapping in a hosted embedding provider later
only means implementing the same `embed`/`embed_batch` interface.

## The index cache's consistency model

`IVFIndex` in the C++ engine is a point-in-time snapshot: it doesn't track
inserts/removes to the store it was built from, and it refuses to search
against a store whose `version()` has moved on (see
`engine/include/vdb/ivf_index.hpp`). `app/index_cache.py` mirrors that at
the API level - any vector mutation (add, update, delete) invalidates the
cached index for that collection, so a stale index is never silently
served. The tradeoff: after bulk changes, you need to call `POST
/collections/{id}/index` again before `use_index=true` works.

## Request flow for a search

```
Dataset → Embedding generation → Vector storage (pgvector)
                                        │
                          ┌─────────────┴─────────────┐
                          │                            │
                 exact (pgvector SQL)         approximate (C++ IVF, cached)
                          │                            │
                          └─────────────┬──────────────┘
                                Filtering / ranking
                                        │
                              Results + latency + used_index
```
