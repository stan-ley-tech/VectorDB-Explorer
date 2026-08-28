# Benchmark methodology

Implemented in [`backend/app/benchmark_runner.py`](../backend/app/benchmark_runner.py).

## What gets measured

Every benchmark run compares two search strategies over the **same**
synthetic dataset and the **same** query set:

- **`brute_force`** - exhaustive search (native C++ engine when available,
  otherwise a NumPy fallback - see `app/engine_bridge.py`). Always exact by
  construction, so it doubles as the ground truth for recall.
- **`indexed`** - the C++ engine's IVF index: vectors are partitioned into
  `nlist` clusters via k-means, and a query only scans the `nprobe` nearest
  clusters. Skipped (not failed) when the native extension isn't built,
  since it has no Python-only equivalent.

For each mode, the runner reports:

| Metric | Meaning |
|---|---|
| `avg_latency_ms` / `p95_latency_ms` / `p99_latency_ms` | per-query wall-clock time |
| `throughput_qps` | queries completed per second across the whole run |
| `recall_at_k` | for `indexed`, the fraction of brute-force's top-k that the index also returned, averaged over all queries. Always `1.0` for `brute_force` |
| `memory_bytes` | resident size of the vector store (+ index, for `indexed`) |
| `index_build_ms` | time to build the IVF index (`null` for `brute_force`) |

## Why synthetic data

A benchmark generates its own vectors and queries (`np.random.default_rng`,
normally distributed, seeded) sized by the request's `dataset_size` and
`dimensions` - it does not read from the collection it's nominally attached
to. That keeps a run reproducible and its cost predictable regardless of
what a given collection happens to contain at the moment; the collection id
in the URL is just where the result gets filed.

## Concurrency

`num_queries` requests are dispatched through a `ThreadPoolExecutor` sized
by `concurrency`. Threads, not asyncio: the search work itself (NumPy or the
native engine) is CPU-bound, not I/O-bound, so threads are the simplest way
to get overlapping in-flight queries for a throughput measurement without
restructuring the search path around async I/O it doesn't do.

## Running a large benchmark outside the API

`POST /collections/{id}/benchmark` runs synchronously inside the HTTP
request, which is fine for a quick check but not for a 100k-vector sweep.
For that, use the standalone CLI, which calls the exact same
`run_benchmark()` function:

```bash
PYTHONPATH=backend python benchmarks/run_benchmark.py \
    --dataset-size 100000 --dimensions 256 --num-queries 1000 --nlist 200 --nprobe 16
```

Results are printed to stdout and saved as JSON under `benchmarks/results/`.
