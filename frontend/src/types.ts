export type MetricName = "cosine" | "euclidean" | "dot";

export interface Collection {
  id: string;
  name: string;
  dim: number;
  metric: MetricName;
  created_at: string;
}

export interface CollectionStats {
  document_count: number;
  dimensions: number;
  metric: MetricName;
  index_type: string | null;
  storage_bytes: number;
}

export interface VectorRecord {
  id: string;
  external_id: string;
  document: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface SearchHit {
  id: string;
  external_id: string;
  document: string | null;
  score: number;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  results: SearchHit[];
  latency_ms: number;
  used_index: boolean;
}

export interface IndexInfo {
  id: string;
  index_type: string;
  metric: MetricName;
  nlist: number | null;
  status: string;
  built_at: string | null;
  created_at: string;
}

export interface BenchmarkConfig {
  dataset_size: number;
  dimensions: number;
  top_k: number;
  metric: MetricName;
  nlist: number;
  nprobe: number;
  num_queries: number;
  concurrency: number;
}

export interface BenchmarkResult {
  mode: "brute_force" | "indexed";
  avg_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  throughput_qps: number;
  recall_at_k: number | null;
  memory_bytes: number;
  index_build_ms: number | null;
}

export interface Benchmark {
  id: string;
  name: string;
  config: BenchmarkConfig;
  status: string;
  created_at: string;
  completed_at: string | null;
  results: BenchmarkResult[];
}

export interface ImportResponse {
  imported: number;
  failed: number;
  errors: string[];
}

export interface ApiErrorBody {
  detail?: string;
}
