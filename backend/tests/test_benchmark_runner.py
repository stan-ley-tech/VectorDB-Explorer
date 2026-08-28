import numpy as np

import app.engine_bridge as engine_bridge
from app.benchmark_runner import (
    _mean_recall_at_k,
    _percentile,
    generate_synthetic_dataset,
    run_benchmark,
)
from app.engine_bridge import Metric
from tests.conftest import requires_native_engine


def test_generate_synthetic_dataset_has_requested_shape():
    vectors = generate_synthetic_dataset(size=50, dim=16)
    assert vectors.shape == (50, 16)
    assert vectors.dtype == np.float32


def test_generate_synthetic_dataset_is_deterministic_for_a_given_seed():
    a = generate_synthetic_dataset(size=20, dim=8, seed=7)
    b = generate_synthetic_dataset(size=20, dim=8, seed=7)
    assert np.array_equal(a, b)


def test_percentile_of_empty_list_is_zero():
    assert _percentile([], 95) == 0.0


def test_percentile_of_single_value_is_that_value():
    assert _percentile([42.0], 99) == 42.0


def test_mean_recall_perfect_match_is_one():
    ground_truth = [[1, 2, 3], [4, 5, 6]]
    candidate = [[1, 2, 3], [4, 5, 6]]
    assert _mean_recall_at_k(ground_truth, candidate, k=3) == 1.0


def test_mean_recall_no_overlap_is_zero():
    ground_truth = [[1, 2, 3]]
    candidate = [[7, 8, 9]]
    assert _mean_recall_at_k(ground_truth, candidate, k=3) == 0.0


def test_mean_recall_partial_overlap():
    ground_truth = [[1, 2, 3, 4]]
    candidate = [[1, 2, 9, 9]]
    # 2 of the 4 ground-truth ids were found
    assert _mean_recall_at_k(ground_truth, candidate, k=4) == 0.5


def test_mean_recall_of_empty_ground_truth_is_zero():
    assert _mean_recall_at_k([], [], k=5) == 0.0


def test_run_benchmark_without_native_engine_returns_brute_force_only(monkeypatch):
    # Forces the same graceful-degradation path a machine without a C++
    # toolchain hits, regardless of whether this environment actually has
    # the compiled extension.
    monkeypatch.setattr(engine_bridge, "NATIVE_ENGINE_AVAILABLE", False)
    results = run_benchmark(
        dataset_size=200,
        dimensions=16,
        top_k=5,
        metric=Metric.COSINE,
        nlist=4,
        nprobe=2,
        num_queries=10,
        concurrency=2,
        seed=1,
    )

    assert len(results) == 1
    brute_force = results[0]
    assert brute_force.mode == "brute_force"
    assert brute_force.recall_at_k == 1.0
    assert brute_force.avg_latency_ms >= 0.0
    assert brute_force.throughput_qps > 0.0
    assert brute_force.memory_bytes > 0
    assert brute_force.index_build_ms is None


def test_run_benchmark_is_consistent_between_sequential_and_concurrent_runs(monkeypatch):
    monkeypatch.setattr(engine_bridge, "NATIVE_ENGINE_AVAILABLE", False)
    kwargs = dict(
        dataset_size=100,
        dimensions=8,
        top_k=3,
        metric=Metric.EUCLIDEAN,
        nlist=4,
        nprobe=2,
        num_queries=15,
        seed=3,
    )
    sequential = run_benchmark(**kwargs, concurrency=1)
    concurrent = run_benchmark(**kwargs, concurrency=4)

    # Same seeds, same data/queries - only the execution strategy differs,
    # so recall must match even though wall-clock timings will vary.
    assert sequential[0].recall_at_k == concurrent[0].recall_at_k


@requires_native_engine
def test_run_benchmark_with_native_engine_includes_indexed_mode():
    results = run_benchmark(
        dataset_size=300,
        dimensions=16,
        top_k=5,
        metric=Metric.COSINE,
        nlist=6,
        nprobe=3,
        num_queries=20,
        concurrency=2,
        seed=5,
    )

    assert {r.mode for r in results} == {"brute_force", "indexed"}
    indexed = next(r for r in results if r.mode == "indexed")
    assert indexed.recall_at_k is not None
    assert 0.0 <= indexed.recall_at_k <= 1.0
    assert indexed.index_build_ms is not None
    assert indexed.memory_bytes > 0
