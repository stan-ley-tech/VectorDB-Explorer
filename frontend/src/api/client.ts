import type {
  Benchmark,
  BenchmarkConfig,
  Collection,
  CollectionStats,
  ImportResponse,
  IndexInfo,
  MetricName,
  SearchResponse,
  VectorRecord,
} from "../types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}) as { detail?: string });
    throw new ApiError(response.status, body.detail ?? `request to ${path} failed with ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function listCollections(): Promise<Collection[]> {
  return request("/collections");
}

export function createCollection(input: { name: string; dim: number; metric: MetricName }): Promise<Collection> {
  return request("/collections", { method: "POST", body: JSON.stringify(input) });
}

export function getCollection(id: string): Promise<Collection> {
  return request(`/collections/${id}`);
}

export function getCollectionStats(id: string): Promise<CollectionStats> {
  return request(`/collections/${id}/stats`);
}

export function deleteCollection(id: string): Promise<void> {
  return request(`/collections/${id}`, { method: "DELETE" });
}

export function upsertVectors(
  collectionId: string,
  vectors: Array<{ external_id: string; document?: string; metadata?: Record<string, unknown> }>,
): Promise<VectorRecord[]> {
  return request(`/collections/${collectionId}/vectors`, {
    method: "POST",
    body: JSON.stringify({ vectors }),
  });
}

export function importDataset(
  collectionId: string,
  file: File,
  format: "csv" | "json",
): Promise<ImportResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request(`/collections/${collectionId}/import?format=${format}`, {
    method: "POST",
    body: formData,
  });
}

export function exportDatasetUrl(collectionId: string, format: "csv" | "json"): string {
  return `${BASE_URL}/collections/${collectionId}/export?format=${format}`;
}

export function searchCollection(
  collectionId: string,
  input: {
    query?: string;
    top_k: number;
    metric?: MetricName;
    filter?: Record<string, unknown>;
    use_index: boolean;
    nprobe?: number;
  },
): Promise<SearchResponse> {
  return request(`/collections/${collectionId}/search`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function buildIndex(collectionId: string, nlist: number): Promise<IndexInfo> {
  return request(`/collections/${collectionId}/index`, {
    method: "POST",
    body: JSON.stringify({ index_type: "ivf", nlist }),
  });
}

export function getIndex(collectionId: string): Promise<IndexInfo | null> {
  return request(`/collections/${collectionId}/index`);
}

export function runBenchmark(collectionId: string, name: string, config: BenchmarkConfig): Promise<Benchmark> {
  return request(`/collections/${collectionId}/benchmark`, {
    method: "POST",
    body: JSON.stringify({ name, config }),
  });
}

export function getBenchmark(id: string): Promise<Benchmark> {
  return request(`/benchmarks/${id}`);
}
