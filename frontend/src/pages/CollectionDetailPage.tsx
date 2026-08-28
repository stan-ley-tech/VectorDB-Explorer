import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, getCollection } from "../api/client";
import BenchmarkPanel from "../components/BenchmarkPanel";
import SearchPanel from "../components/SearchPanel";
import StatsPanel from "../components/StatsPanel";
import type { Collection } from "../types";

type Tab = "overview" | "search" | "benchmark";

export default function CollectionDetailPage() {
  const { collectionId } = useParams<{ collectionId: string }>();
  const [collection, setCollection] = useState<Collection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");

  useEffect(() => {
    if (!collectionId) return;
    getCollection(collectionId)
      .then(setCollection)
      .catch((err) => setError(err instanceof ApiError ? err.message : "failed to load collection"));
  }, [collectionId]);

  if (error) {
    return (
      <div>
        <p className="error-text">{error}</p>
        <Link to="/collections">Back to collections</Link>
      </div>
    );
  }

  if (!collection || !collectionId) {
    return <p className="empty-state">Loading...</p>;
  }

  return (
    <div>
      <p>
        <Link to="/collections">Collections</Link> / <strong>{collection.name}</strong>
      </p>

      <div className="tabs">
        <button className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")}>
          Overview
        </button>
        <button className={tab === "search" ? "active" : ""} onClick={() => setTab("search")}>
          Search
        </button>
        <button className={tab === "benchmark" ? "active" : ""} onClick={() => setTab("benchmark")}>
          Benchmark
        </button>
      </div>

      {tab === "overview" && <StatsPanel collectionId={collectionId} />}
      {tab === "search" && <SearchPanel collection={collection} />}
      {tab === "benchmark" && <BenchmarkPanel collection={collection} />}
    </div>
  );
}
