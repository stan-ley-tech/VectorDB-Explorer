import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, createCollection, listCollections } from "../api/client";
import type { Collection, MetricName } from "../types";

export default function CollectionsPage() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [dim, setDim] = useState(256);
  const [metric, setMetric] = useState<MetricName>("cosine");
  const [creating, setCreating] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      setCollections(await listCollections());
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "failed to load collections");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function handleCreate(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    try {
      await createCollection({ name: name.trim(), dim, metric });
      setName("");
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "failed to create collection");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div>
      <div className="card">
        <h2>New collection</h2>
        <form onSubmit={handleCreate} className="form-row">
          <div className="form-field">
            <label htmlFor="name">Name</label>
            <input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="docs" />
          </div>
          <div className="form-field">
            <label htmlFor="dim">Dimensions</label>
            <input
              id="dim"
              type="number"
              min={1}
              value={dim}
              onChange={(e) => setDim(Number(e.target.value))}
              style={{ width: 100 }}
            />
          </div>
          <div className="form-field">
            <label htmlFor="metric">Metric</label>
            <select id="metric" value={metric} onChange={(e) => setMetric(e.target.value as MetricName)}>
              <option value="cosine">cosine</option>
              <option value="euclidean">euclidean</option>
              <option value="dot">dot</option>
            </select>
          </div>
          <button type="submit" className="primary" disabled={creating || !name.trim()}>
            {creating ? "Creating..." : "Create collection"}
          </button>
        </form>
      </div>

      {error && (
        <p className="error-text" style={{ marginTop: 12 }}>
          {error}
        </p>
      )}

      <div className="card">
        <h2>Collections</h2>
        {loading ? (
          <p className="empty-state">Loading...</p>
        ) : collections.length === 0 ? (
          <p className="empty-state">No collections yet. Create one above to get started.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Dimensions</th>
                <th>Metric</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {collections.map((collection) => (
                <tr key={collection.id}>
                  <td>
                    <Link to={`/collections/${collection.id}`}>{collection.name}</Link>
                  </td>
                  <td className="mono">{collection.dim}</td>
                  <td className="mono">{collection.metric}</td>
                  <td className="mono">{new Date(collection.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
