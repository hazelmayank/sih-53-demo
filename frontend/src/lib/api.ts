import type { AppConfig, CellDetail, ClassInfo } from "../types";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body?: unknown) =>
  req<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  classes: () => req<ClassInfo[]>("/api/classes"),
  config: () => req<AppConfig>("/api/config"),
  patchConfig: (patch: unknown) => post<AppConfig>("/api/config", patch),
  resetConfig: () => post<AppConfig>("/api/config/reset"),
  start: () => post("/api/scene/start"),
  stop: () => post("/api/scene/stop"),
  step: () => post("/api/scene/step"),
  reset: () => post("/api/scene/reset"),
  seek: (frame_index: number) => post("/api/scene/seek", { frame_index }),
  presets: () => req<{ id: string; label: string; description: string }[]>("/api/scene/presets"),
  files: () => req<{ path: string; kind: string; frames: number }[]>("/api/scene/files"),
  cell: (frameId: number, level: number, ix: number, iy: number) =>
    req<CellDetail>(`/api/frame/${frameId}/cell?level=${level}&ix=${ix}&iy=${iy}`),
  benchmark: () => req<BenchmarkResponse>("/api/benchmark?series=0"),
  runBenchmark: (frames: number, scaling: boolean) => post("/api/benchmark/run", { frames, scaling }),
  upload: async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/scene/upload", { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail ?? res.statusText);
    return body as { path: string; points: number };
  },
};

export interface Stat {
  median: number;
  p95: number;
  mean: number;
  min: number;
  max: number;
}

export interface BenchmarkResult {
  created: string;
  duration_s: number;
  frames: number;
  config: Record<string, unknown>;
  summary: {
    points: Stat;
    latency_ms: Stat;
    adaptive_projection_ms: Stat;
    uniform_projection_ms: Stat;
    adaptive_cells: Stat;
    uniform_sparse_cells: Stat;
    uniform_dense_cells: number;
    adaptive_mb: Stat;
    uniform_sparse_mb: Stat;
    uniform_dense_mb: number;
    reduction_vs_dense_pct: Stat;
    reduction_vs_sparse_pct: Stat;
    partition_ok_all: boolean;
    seg_accuracy: Stat | null;
  };
  per_frame: {
    frame: number;
    points: number;
    latency_ms: number;
    adaptive_projection_ms: number;
    uniform_projection_ms: number;
    adaptive_cells: number;
    uniform_sparse_cells: number;
  }[];
  scaling: {
    beams: number;
    points: number;
    latency_ms: number;
    adaptive_projection_ms: number;
    uniform_projection_ms: number;
    adaptive_cells: number;
    uniform_sparse_cells: number;
  }[];
  saved_to?: string;
}

export interface BenchmarkResponse {
  live: { summary: Record<string, Stat | number>; latest: unknown };
  run: { running: boolean; progress: number; message: string; result: BenchmarkResult | null; error?: string | null };
}
