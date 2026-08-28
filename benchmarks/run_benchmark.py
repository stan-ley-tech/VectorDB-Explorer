#!/usr/bin/env python3
"""Standalone benchmark runner for configurations too large or slow to run
through the HTTP API (POST /collections/{id}/benchmark runs synchronously
inside a request, so it's fine for quick interactive checks but not for a
100k-vector, 5000-query sweep). Reuses the exact same brute-force-vs-IVF
comparison logic from app.benchmark_runner - this script is just a CLI
wrapper around it that isn't bound by a request timeout.

Usage:
    pip install -r backend/requirements.txt
    PYTHONPATH=backend python benchmarks/run_benchmark.py \\
        --dataset-size 100000 --dimensions 256 --num-queries 1000

Results are printed to stdout and saved as JSON under benchmarks/results/.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.benchmark_runner import run_benchmark  # noqa: E402
from app.engine_bridge import NATIVE_ENGINE_AVAILABLE, Metric  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-size", type=int, default=50_000)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--metric", choices=["cosine", "euclidean", "dot"], default="cosine")
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="defaults to benchmarks/results/<timestamp>.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not NATIVE_ENGINE_AVAILABLE:
        print(
            "warning: vdb_native is not importable - only brute-force results will be produced. "
            "Build engine/ with -DVDB_BUILD_PYTHON_BINDINGS=ON and put it on PYTHONPATH to include "
            "the indexed comparison.",
            file=sys.stderr,
        )

    results = run_benchmark(
        dataset_size=args.dataset_size,
        dimensions=args.dimensions,
        top_k=args.top_k,
        metric=Metric(args.metric),
        nlist=args.nlist,
        nprobe=args.nprobe,
        num_queries=args.num_queries,
        concurrency=args.concurrency,
        seed=args.seed,
    )

    for result in results:
        recall = f"{result.recall_at_k:.3f}" if result.recall_at_k is not None else "n/a"
        print(
            f"{result.mode:12s} avg={result.avg_latency_ms:8.2f}ms  p95={result.p95_latency_ms:8.2f}ms  "
            f"p99={result.p99_latency_ms:8.2f}ms  throughput={result.throughput_qps:9.1f} qps  "
            f"recall@{args.top_k}={recall}  memory={result.memory_bytes / (1024 * 1024):.1f}MB"
        )

    output_path = args.output or (
        Path(__file__).resolve().parent / "results" / f"benchmark-{int(datetime.now(timezone.utc).timestamp())}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "dataset_size": args.dataset_size,
            "dimensions": args.dimensions,
            "top_k": args.top_k,
            "metric": args.metric,
            "nlist": args.nlist,
            "nprobe": args.nprobe,
            "num_queries": args.num_queries,
            "concurrency": args.concurrency,
            "seed": args.seed,
        },
        "run_at": datetime.now(timezone.utc).isoformat(),
        "results": [result.__dict__ for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
