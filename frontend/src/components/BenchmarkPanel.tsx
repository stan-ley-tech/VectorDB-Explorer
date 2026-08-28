import { useState } from "react";
import { ApiError, runBenchmark } from "../api/client";
import { formatMs, formatNumber } from "../format";
import type { Benchmark, BenchmarkConfig, Collection, MetricName } from "../types";

const defaultConfig: BenchmarkConfig = {
  dataset_size: 5000,
  dimensions: 256,
  top_k: 10,
  metric: "cosine",
  nlist: 100,
  nprobe: 8,
  num_queries: 200,
  concurrency: 4,
};

export default function BenchmarkPanel({ collection }: { collection: Collection }) {
  const [config, setConfig] = useState<BenchmarkConfig>({ ...defaultConfig, metric: collection.metric });
  const [running, setRunning] = useState(false);
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const [error, setError] = useState<string | null>(null);

  function set<K extends keyof BenchmarkConfig>(key: K, value: BenchmarkConfig[K]) {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }

  async function handleRun(event: React.FormEvent) {
    event.preventDefault();
    setRunning(true);
    setError(null);
    try {
      const result = await runBenchmark(collection.id, `${collection.name} benchmark`, config);
      setBenchmark(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "benchmark failed");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div>
      <form onSubmit={handleRun} className="card">
        <h3>Configuration</h3>
        <div className="form-row">
          <NumberField label="Dataset size" value={config.dataset_size} onChange={(v) => set("dataset_size", v)} />
          <NumberField label="Dimensions" value={config.dimensions} onChange={(v) => set("dimensions", v)} />
          <NumberField label="Top-K" value={config.top_k} onChange={(v) => set("top_k", v)} />
          <div className="form-field">
            <label htmlFor="metric">Metric</label>
            <select id="metric" value={config.metric} onChange={(e) => set("metric", e.target.value as MetricName)}>
              <option value="cosine">cosine</option>
              <option value="euclidean">euclidean</option>
              <option value="dot">dot</option>
            </select>
          </div>
        </div>
        <div className="form-row" style={{ marginTop: 12 }}>
          <NumberField label="nlist" value={config.nlist} onChange={(v) => set("nlist", v)} />
          <NumberField label="nprobe" value={config.nprobe} onChange={(v) => set("nprobe", v)} />
          <NumberField label="Queries" value={config.num_queries} onChange={(v) => set("num_queries", v)} />
          <NumberField label="Concurrency" value={config.concurrency} onChange={(v) => set("concurrency", v)} />
          <button type="submit" className="primary" disabled={running}>
            {running ? "Running..." : "Run benchmark"}
          </button>
        </div>
      </form>

      {error && (
        <p className="error-text" style={{ marginTop: 12 }}>
          {error}
        </p>
      )}

      {benchmark && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Results — {benchmark.status}</h3>
          <table>
            <thead>
              <tr>
                <th>Mode</th>
                <th>Avg latency</th>
                <th>P95</th>
                <th>P99</th>
                <th>Throughput</th>
                <th>Recall@{config.top_k}</th>
                <th>Memory</th>
                <th>Build time</th>
              </tr>
            </thead>
            <tbody>
              {benchmark.results.map((result) => (
                <tr key={result.mode}>
                  <td>{result.mode === "brute_force" ? "Brute force" : "Indexed (IVF)"}</td>
                  <td className="mono">{formatMs(result.avg_latency_ms)}</td>
                  <td className="mono">{formatMs(result.p95_latency_ms)}</td>
                  <td className="mono">{formatMs(result.p99_latency_ms)}</td>
                  <td className="mono">{formatNumber(Math.round(result.throughput_qps))} qps</td>
                  <td className="mono">
                    {result.recall_at_k !== null ? `${(result.recall_at_k * 100).toFixed(1)}%` : "-"}
                  </td>
                  <td className="mono">{formatNumber(Math.round(result.memory_bytes / 1024))} KB</td>
                  <td className="mono">{result.index_build_ms !== null ? formatMs(result.index_build_ms) : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {benchmark.results.length === 1 && (
            <p style={{ color: "var(--text-dim)", marginTop: 8 }}>
              Only brute-force results are shown because the native engine extension (vdb_native) isn't built in
              this environment. See engine/README for the build command.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  const id = label.toLowerCase().replace(/\s+/g, "-");
  return (
    <div className="form-field">
      <label htmlFor={id}>{label}</label>
      <input id={id} type="number" min={1} value={value} onChange={(e) => onChange(Number(e.target.value))} style={{ width: 100 }} />
    </div>
  );
}
