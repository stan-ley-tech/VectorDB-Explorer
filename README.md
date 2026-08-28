# VectorDB Explorer

A developer tool for inspecting, querying, benchmarking, and managing vector
collections: upload a dataset, generate embeddings, run semantic search,
filter by metadata, and compare brute-force search against an approximate
index on latency, recall, memory, and throughput.

It's built around a small hand-written C++ vector search engine rather than
leaning entirely on pgvector - see [why](#why-a-custom-c-engine) below.

[![CI](https://github.com/stan-ley-tech/VectorDB-Explorer/actions/workflows/ci.yml/badge.svg)](https://github.com/stan-ley-tech/VectorDB-Explorer/actions/workflows/ci.yml)

## Stack

C++17 · Python (FastAPI) · React/TypeScript · PostgreSQL + pgvector · Docker

## Quickstart

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- API: http://localhost:8000 (interactive docs at `/docs`)

Create a collection, import [`datasets/sample_docs.csv`](datasets/sample_docs.csv)
from the collection's Overview tab, then try a search from the Search tab.

## What it does

- Create collections with a chosen dimension and distance metric
- Upload CSV/JSON datasets (streamed and imported in batches, not loaded
  fully into memory)
- Generate embeddings locally with no API key or GPU required (see
  [embeddings](#why-a-hashing-trick-embedder))
- Search by semantic similarity, with optional metadata filtering
- Compare exact (pgvector) search against an approximate (IVF) index built
  by the C++ engine, side by side
- Inspect vector dimensions, document counts, and estimated storage per
  collection
- Update or delete individual vectors; import/export whole datasets
- Configure and run benchmarks (dataset size, dimensions, top-K, metric,
  index config, query count, concurrency) and see latency percentiles,
  throughput, recall@k, and memory usage

## Architecture

```
Dataset (CSV/JSON)
   |
   v
Embedding generation  (backend/app/embeddings.py)
   |
   v
Vector storage         (Postgres + pgvector, unconstrained dimension)
   |
   +----------------------------+
   |                            |
   v                            v
Exact search (pgvector SQL)   Approximate search (C++ IVF index, cached)
   |                            |
   +----------------------------+
                |
                v
     Filtering / ranking / results + latency
```

Full write-up, including why the schema doesn't use pgvector's typed
`vector(n)` column and how the two search paths stay consistent with each
other: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository layout

```
vectordb-explorer/
├── engine/              C++17 search engine: distance metrics, brute-force
│   ├── include/vdb/     top-K, IVF approximate index. Own test suite,
│   ├── cpp/             own README (engine/README.md).
│   ├── bindings/        pybind11 module (vdb_native)
│   └── tests/
├── backend/             FastAPI service: API, Postgres schema (SQLAlchemy +
│   ├── app/             Alembic), dataset import/export, benchmark runner.
│   └── tests/           Falls back to a NumPy brute-force implementation
│                        when vdb_native isn't built.
├── frontend/             React + TypeScript dashboard (Vite). Unit tests
│   ├── src/              with Vitest, E2E with Playwright.
│   └── e2e/
├── benchmarks/          Standalone CLI for large offline benchmark runs
├── datasets/            Sample CSV/JSON datasets
├── docs/                Architecture, API reference, benchmark methodology
├── docker-compose.yml
├── Dockerfile           Backend image (multi-stage: builds the C++ engine,
│                        then a slim Python runtime)
└── .github/workflows/   CI: engine, backend, frontend, e2e, Docker builds
```

## Local development

Each piece can be run and tested independently.

### Engine (C++)

```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
ctest --test-dir engine/build --output-on-failure
```

See [engine/README.md](engine/README.md) for building the Python bindings.

### Backend (Python)

Requires a running Postgres with the `vector` extension (`docker compose up
postgres` is the easiest way to get one).

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
export DATABASE_URL=postgresql+psycopg://vectordb:vectordb@localhost:5432/vectordb
alembic upgrade head
uvicorn app.main:app --reload
```

Run the tests:

```bash
pytest                 # unit tests always run; DB/native-engine tests skip
                        # gracefully if Postgres/vdb_native aren't available
```

### Frontend (React/TypeScript)

```bash
cd frontend
npm install
npm run dev             # dev server
npm run test             # Vitest unit/component tests
npm run test:e2e         # Playwright, needs a running backend (see playwright.config.ts)
npm run build             # type-checks and produces dist/
```

## API

Quick reference in [docs/API.md](docs/API.md); full interactive schema at
`/docs` on a running backend.

```
POST   /collections
GET    /collections/{id}
GET    /collections/{id}/stats
POST   /collections/{id}/vectors
POST   /collections/{id}/import
GET    /collections/{id}/export
POST   /collections/{id}/search
POST   /collections/{id}/index
POST   /collections/{id}/benchmark
GET    /benchmarks/{id}
```

## Testing

| Layer | Tool | What's covered |
|---|---|---|
| Engine | doctest | distance math, vector store mutation, brute-force ranking, IVF build/search correctness, staleness checks |
| Backend | pytest | embeddings, ingestion, engine bridge (+ fallback), index cache, benchmark runner, full API against real Postgres |
| Frontend | Vitest + Testing Library | dashboard pages and panels, mocked API |
| End-to-end | Playwright | create collection → import → search → benchmark, against a real backend |

CI (`.github/workflows/ci.yml`) runs all four, plus a check that both Docker
images build, on every push.

## Why a custom C++ engine

pgvector is the storage layer, but the brief for this project asked for a
hand-rolled comparison between brute-force and indexed search rather than
leaning on pgvector's own indexing. `engine/` implements cosine/Euclidean/dot
distance, an O(n·dim) brute-force top-K search with a bounded min-heap
(O(candidates·log k) instead of a full sort), and a from-scratch IVF index
(k-means++ initialization, Lloyd's iteration, cluster-probing search). The
benchmark endpoint runs both over the same data and queries so the
comparison is direct, not two different systems measured under different
conditions. Details: [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## Why a hashing-trick embedder

`backend/app/embeddings.py` generates embeddings locally with a
deterministic hashing-trick vectorizer (tokens/bigrams hashed into buckets,
L2-normalized) instead of calling out to a hosted model. It's not a
substitute for a trained model's semantics, but the whole system - storage,
indexing, search, benchmarking, CI - runs offline with no API key, no GPU,
and no network dependency. Swapping in a hosted provider later only means
implementing the same `embed`/`embed_batch` interface.

## License

[MIT](LICENSE)
