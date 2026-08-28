import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.benchmark_runner import run_benchmark
from app.database import get_db
from app.engine_bridge import Metric
from app.models import Benchmark, BenchmarkResult
from app.routers.collections import get_collection_or_404
from app.schemas import BenchmarkCreate, BenchmarkOut, BenchmarkResultOut

router = APIRouter(tags=["benchmarks"])


def _to_out(benchmark: Benchmark) -> BenchmarkOut:
    return BenchmarkOut(
        id=benchmark.id,
        name=benchmark.name,
        config=benchmark.config,
        status=benchmark.status,
        created_at=benchmark.created_at,
        completed_at=benchmark.completed_at,
        results=[
            BenchmarkResultOut(
                mode=r.mode,
                avg_latency_ms=r.avg_latency_ms,
                p95_latency_ms=r.p95_latency_ms,
                p99_latency_ms=r.p99_latency_ms,
                throughput_qps=r.throughput_qps,
                recall_at_k=r.recall_at_k,
                memory_bytes=r.memory_bytes,
                index_build_ms=r.index_build_ms,
            )
            for r in benchmark.results
        ],
    )


@router.post("/collections/{collection_id}/benchmark", response_model=BenchmarkOut, status_code=201)
def run_collection_benchmark(
    collection_id: uuid.UUID, payload: BenchmarkCreate, db: Session = Depends(get_db)
) -> BenchmarkOut:
    """Runs brute-force vs. indexed search over a synthetic dataset sized by
    `payload.config`, not the collection's real contents - see
    app.benchmark_runner for why. The collection only anchors where the
    result is filed and which real-world scale the config is imitating.

    This runs synchronously in the request. The config caps (dataset_size,
    num_queries) keep worst-case runtime bounded, but a very large benchmark
    is still a slow HTTP request - see benchmarks/run_benchmark.py for an
    offline alternative that isn't bound by a request timeout.
    """
    get_collection_or_404(db, collection_id)

    benchmark = Benchmark(
        collection_id=collection_id, name=payload.name, config=payload.config.model_dump(), status="running"
    )
    db.add(benchmark)
    db.commit()
    db.refresh(benchmark)

    try:
        mode_results = run_benchmark(
            dataset_size=payload.config.dataset_size,
            dimensions=payload.config.dimensions,
            top_k=payload.config.top_k,
            metric=Metric(payload.config.metric),
            nlist=payload.config.nlist,
            nprobe=payload.config.nprobe,
            num_queries=payload.config.num_queries,
            concurrency=payload.config.concurrency,
        )
    except Exception:
        benchmark.status = "failed"
        db.commit()
        raise

    for mode_result in mode_results:
        db.add(
            BenchmarkResult(
                benchmark_id=benchmark.id,
                mode=mode_result.mode,
                avg_latency_ms=mode_result.avg_latency_ms,
                p95_latency_ms=mode_result.p95_latency_ms,
                p99_latency_ms=mode_result.p99_latency_ms,
                throughput_qps=mode_result.throughput_qps,
                recall_at_k=mode_result.recall_at_k,
                memory_bytes=mode_result.memory_bytes,
                index_build_ms=mode_result.index_build_ms,
            )
        )

    benchmark.status = "completed"
    benchmark.completed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(benchmark)

    return _to_out(benchmark)


@router.get("/benchmarks/{benchmark_id}", response_model=BenchmarkOut)
def get_benchmark(benchmark_id: uuid.UUID, db: Session = Depends(get_db)) -> BenchmarkOut:
    benchmark = db.get(Benchmark, benchmark_id)
    if benchmark is None:
        raise HTTPException(status_code=404, detail="benchmark not found")
    return _to_out(benchmark)
