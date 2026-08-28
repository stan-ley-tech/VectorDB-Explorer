import { useEffect, useState } from "react";
import { ApiError, buildIndex, exportDatasetUrl, getCollectionStats, getIndex, importDataset } from "../api/client";
import { formatBytes, formatNumber } from "../format";
import type { CollectionStats, IndexInfo } from "../types";

export default function StatsPanel({ collectionId }: { collectionId: string }) {
  const [stats, setStats] = useState<CollectionStats | null>(null);
  const [index, setIndex] = useState<IndexInfo | null>(null);
  const [nlist, setNlist] = useState(100);
  const [building, setBuilding] = useState(false);
  const [importing, setImporting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      const [statsResult, indexResult] = await Promise.all([
        getCollectionStats(collectionId),
        getIndex(collectionId),
      ]);
      setStats(statsResult);
      setIndex(indexResult);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "failed to load collection stats");
    }
  }

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collectionId]);

  async function handleBuildIndex() {
    setBuilding(true);
    setMessage(null);
    try {
      await buildIndex(collectionId, nlist);
      setMessage(`Index built with nlist=${nlist}.`);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "failed to build index");
    } finally {
      setBuilding(false);
    }
  }

  async function handleImport(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    const format = file.name.endsWith(".json") ? "json" : "csv";
    setImporting(true);
    setMessage(null);
    try {
      const result = await importDataset(collectionId, file, format);
      setMessage(`Imported ${result.imported} record(s), ${result.failed} failed.`);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "import failed");
    } finally {
      setImporting(false);
    }
  }

  if (error) return <p className="error-text">{error}</p>;
  if (!stats) return <p className="empty-state">Loading...</p>;

  return (
    <div>
      <div className="stat-grid">
        <div className="stat-tile">
          <div className="label">Documents</div>
          <div className="value">{formatNumber(stats.document_count)}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Dimensions</div>
          <div className="value">{formatNumber(stats.dimensions)}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Index</div>
          <div className="value">{stats.index_type ?? "none"}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Storage (est.)</div>
          <div className="value">{formatBytes(stats.storage_bytes)}</div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Approximate index</h3>
        {index && (
          <p className="mono" style={{ color: "var(--text-dim)" }}>
            status: {index.status} · nlist: {index.nlist ?? "-"} · built:{" "}
            {index.built_at ? new Date(index.built_at).toLocaleString() : "never"}
          </p>
        )}
        <div className="form-row">
          <div className="form-field">
            <label htmlFor="nlist">nlist</label>
            <input
              id="nlist"
              type="number"
              min={1}
              value={nlist}
              onChange={(e) => setNlist(Number(e.target.value))}
              style={{ width: 100 }}
            />
          </div>
          <button onClick={handleBuildIndex} disabled={building}>
            {building ? "Building..." : "Build IVF index"}
          </button>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Import / export</h3>
        <div className="form-row">
          <div className="form-field">
            <label htmlFor="import-file">Import CSV or JSON</label>
            <input id="import-file" type="file" accept=".csv,.json" onChange={handleImport} disabled={importing} />
          </div>
          <a href={exportDatasetUrl(collectionId, "json")}>Export JSON</a>
          <a href={exportDatasetUrl(collectionId, "csv")}>Export CSV</a>
        </div>
        {message && <p style={{ marginTop: 8 }}>{message}</p>}
      </div>
    </div>
  );
}
