# API reference

Full interactive docs (request/response schemas, try-it-out) are served by
the running backend at `/docs` (Swagger UI) and `/redoc`. This is a quick
summary of what's available.

## Collections

| Method | Path | Description |
|---|---|---|
| `POST` | `/collections` | Create a collection (`name`, `dim`, `metric`) |
| `GET` | `/collections` | List collections |
| `GET` | `/collections/{id}` | Get a collection |
| `GET` | `/collections/{id}/stats` | Document count, dimensions, index status, estimated storage |
| `DELETE` | `/collections/{id}` | Delete a collection and its vectors |

## Vectors

| Method | Path | Description |
|---|---|---|
| `POST` | `/collections/{id}/vectors` | Upsert vectors by `external_id`. Omit `embedding` to generate one from `document` |
| `GET` | `/collections/{id}/vectors/{vector_id}` | Get one vector |
| `PATCH` | `/collections/{id}/vectors/{vector_id}` | Update document / embedding / metadata |
| `DELETE` | `/collections/{id}/vectors/{vector_id}` | Delete a vector |
| `POST` | `/collections/{id}/import?format=csv\|json` | Streaming dataset import (multipart file upload) |
| `GET` | `/collections/{id}/export?format=csv\|json` | Streaming dataset export |

## Search & indexing

| Method | Path | Description |
|---|---|---|
| `POST` | `/collections/{id}/search` | `query` (text) or `query_vector`, `top_k`, `metric`, `filter` (JSONB containment), `use_index` |
| `POST` | `/collections/{id}/index` | Build an IVF index (`nlist`) over the collection's current vectors |
| `GET` | `/collections/{id}/index` | Latest index status |

## Benchmarks

| Method | Path | Description |
|---|---|---|
| `POST` | `/collections/{id}/benchmark` | Run brute-force vs. indexed comparison over a synthetic dataset (see [docs/BENCHMARKING.md](BENCHMARKING.md)) |
| `GET` | `/benchmarks/{id}` | Fetch a benchmark's results |

## Notes

- `metric` is one of `cosine`, `euclidean`, `dot` and defaults to the
  collection's metric.
- `filter` on `/search` is a JSONB containment match (e.g. `{"category":
  "security"}`) against per-vector metadata; it's only supported on the
  exact (pgvector) path - see [docs/ARCHITECTURE.md](ARCHITECTURE.md) for why.
- Every search is logged (latency, metric, whether the index was used) for
  later inspection via the `queries` table.
