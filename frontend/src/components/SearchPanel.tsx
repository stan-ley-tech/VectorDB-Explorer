import { useState } from "react";
import { ApiError, searchCollection } from "../api/client";
import { formatMs } from "../format";
import type { Collection, SearchHit } from "../types";

export default function SearchPanel({ collection }: { collection: Collection }) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [useIndex, setUseIndex] = useState(false);
  const [results, setResults] = useState<SearchHit[]>([]);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [usedIndex, setUsedIndex] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await searchCollection(collection.id, {
        query: query.trim(),
        top_k: topK,
        use_index: useIndex,
      });
      setResults(response.results);
      setLatencyMs(response.latency_ms);
      setUsedIndex(response.used_index);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "search failed");
    } finally {
      setLoading(false);
    }
  }

  const maxScore = results.length > 0 ? Math.max(...results.map((r) => Math.abs(r.score)), 1e-9) : 1;

  return (
    <div>
      <form onSubmit={handleSearch} className="form-row">
        <div className="form-field" style={{ flex: 1, minWidth: 260 }}>
          <label htmlFor="query">Query</label>
          <input
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="How do I authenticate an API request?"
          />
        </div>
        <div className="form-field">
          <label htmlFor="top-k">Top K</label>
          <input
            id="top-k"
            type="number"
            min={1}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            style={{ width: 80 }}
          />
        </div>
        <div className="form-field">
          <label htmlFor="use-index">
            <input
              id="use-index"
              type="checkbox"
              checked={useIndex}
              onChange={(e) => setUseIndex(e.target.checked)}
            />{" "}
            Use ANN index
          </label>
        </div>
        <button type="submit" className="primary" disabled={loading || !query.trim()}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && (
        <p className="error-text" style={{ marginTop: 12 }}>
          {error}
        </p>
      )}

      {latencyMs !== null && (
        <p className="mono" style={{ color: "var(--text-dim)", marginTop: 12 }}>
          {results.length} result(s) in {formatMs(latencyMs)} ·{" "}
          <span className={`badge ${usedIndex ? "good" : ""}`}>{usedIndex ? "indexed" : "exact (pgvector)"}</span>
        </p>
      )}

      {results.length > 0 && (
        <ol style={{ listStyle: "none", padding: 0, marginTop: 12 }}>
          {results.map((hit, i) => (
            <li key={hit.id} className="card" style={{ marginTop: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                <strong>
                  {i + 1}. {hit.document ?? hit.external_id}
                </strong>
                <span className="mono">{hit.score.toFixed(3)}</span>
              </div>
              <div className="score-bar-track" style={{ marginTop: 8 }}>
                <div
                  className="score-bar-fill"
                  style={{ width: `${Math.max(0, Math.min(1, Math.abs(hit.score) / maxScore)) * 100}%` }}
                />
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
