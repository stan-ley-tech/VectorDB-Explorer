"""Benchmark methodology: brute-force vs. IVF-indexed search compared
head-to-head over the same synthetic dataset and query set.

Vectors and queries are generated synthetically (not pulled from a real
collection) so a benchmark run is reproducible and its cost is controlled
directly by the requested dataset_size/dimensions/num_queries, regardless
of how much data happens to live in any particular collection.

Recall@k is computed against the brute-force run as ground truth, so the
indexed run's recall reported here is always relative to what exhaustive
search actually returned for these queries - not to some external notion
of relevance.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.engine_bridge import (
    Metric,
    NativeEngineUnavailableError,
    SearchHit,
    brute_force_search,
    build_ivf_index,
    estimate_memory_bytes,
)


@dataclass
class ModeResult:
    mode: str
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    recall_at_k: float | None
    memory_bytes: int
    index_build_ms: float | None


def generate_synthetic_dataset(size: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(size, dim)).astype(np.float32)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def _run_queries(
    search_fn: Callable[[np.ndarray, int], list[SearchHit]],
    queries: np.ndarray,
    top_k: int,
    concurrency: int,
) -> tuple[list[list[int]], list[float], float]:
    """Runs every row of `queries` through `search_fn`, timing each call.

    Concurrency uses a thread pool rather than asyncio: the search work
    itself is CPU-bound (numpy or the native engine), so threads are the
    simplest way to get overlapping in-flight queries for a throughput
    measurement without restructuring the search path around async I/O it
    doesn't actually do.
    """
    latencies: list[float] = []
    result_rows: list[list[int]] = []

    def run_one(query: np.ndarray) -> tuple[list[int], float]:
        start = time.perf_counter()
        hits = search_fn(query, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return [h.row for h in hits], elapsed_ms

    start_all = time.perf_counter()
    if concurrency <= 1:
        for query in queries:
            rows, elapsed = run_one(query)
            result_rows.append(rows)
            latencies.append(elapsed)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for rows, elapsed in pool.map(run_one, queries):
                result_rows.append(rows)
                latencies.append(elapsed)
    total_elapsed_s = time.perf_counter() - start_all

    return result_rows, latencies, total_elapsed_s


def _mean_recall_at_k(ground_truth: list[list[int]], candidate: list[list[int]], k: int) -> float:
    if not ground_truth:
        return 0.0
    scores = []
    for truth_rows, candidate_rows in zip(ground_truth, candidate):
        truth_set = set(truth_rows[:k])
        if not truth_set:
            continue
        candidate_set = set(candidate_rows[:k])
        scores.append(len(truth_set & candidate_set) / len(truth_set))
    return float(np.mean(scores)) if scores else 0.0


def run_benchmark(
    dataset_size: int,
    dimensions: int,
    top_k: int,
    metric: Metric,
    nlist: int,
    nprobe: int,
    num_queries: int,
    concurrency: int,
    seed: int = 0,
) -> list[ModeResult]:
    vectors = generate_synthetic_dataset(dataset_size, dimensions, seed=seed)
    queries = np.random.default_rng(seed + 1).normal(size=(num_queries, dimensions)).astype(np.float32)

    brute_rows, brute_latencies, brute_total_s = _run_queries(
        lambda q, k: brute_force_search(vectors, q.tolist(), k, metric), queries, top_k, concurrency
    )
    results = [
        ModeResult(
            mode="brute_force",
            avg_latency_ms=float(np.mean(brute_latencies)) if brute_latencies else 0.0,
            p95_latency_ms=_percentile(brute_latencies, 95),
            p99_latency_ms=_percentile(brute_latencies, 99),
            throughput_qps=(len(queries) / brute_total_s) if brute_total_s > 0 else 0.0,
            recall_at_k=1.0,  # brute force is the ground truth by definition
            memory_bytes=estimate_memory_bytes(vectors),
            index_build_ms=None,
        )
    ]

    try:
        build_start = time.perf_counter()
        index_handle = build_ivf_index(vectors, metric, nlist)
        build_ms = (time.perf_counter() - build_start) * 1000.0

        indexed_rows, indexed_latencies, indexed_total_s = _run_queries(
            lambda q, k: index_handle.search(q.tolist(), k, nprobe), queries, top_k, concurrency
        )

        results.append(
            ModeResult(
                mode="indexed",
                avg_latency_ms=float(np.mean(indexed_latencies)) if indexed_latencies else 0.0,
                p95_latency_ms=_percentile(indexed_latencies, 95),
                p99_latency_ms=_percentile(indexed_latencies, 99),
                throughput_qps=(len(queries) / indexed_total_s) if indexed_total_s > 0 else 0.0,
                recall_at_k=_mean_recall_at_k(brute_rows, indexed_rows, top_k),
                memory_bytes=index_handle.memory_usage_bytes(),
                index_build_ms=build_ms,
            )
        )
    except NativeEngineUnavailableError:
        # No compiled engine in this environment: report brute-force-only
        # results instead of failing the whole benchmark run.
        pass

    return results
