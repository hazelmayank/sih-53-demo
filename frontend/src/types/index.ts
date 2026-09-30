// Shared types. Coordinate convention everywhere: vehicle frame, origin on the ground below
// the LiDAR, x = forward, y = left, z = up (metres).

export interface ClassInfo {
  id: number;
  name: string;
  label: string;
  color: string;
  description: string;
}

export interface Band {
  upper_m: number;
  cell_size_m: number;
}

export interface AppConfig {
  scene: {
    source: "synthetic" | "file";
    preset: string;
    seed: number;
    beams: number;
    azimuth_resolution_deg: number;
    sensor_height_m: number;
    range_noise_m: number;
    dropout: number;
    outlier_ratio: number;
    file_sequence: string | null;
    import_z_offset_m: number;
  };
  processing: {
    min_range_m: number;
    max_range_m: number;
    outlier_filter: boolean;
    voxel_downsample_m: number;
    ground_sectors: number;
    ground_max_slope: number;
    obstacle_height_m: number;
    cluster_cell_m: number;
    min_cluster_points: number;
    segmenter: "geometric_fallback" | "oracle_gt" | "torchscript";
    model_path: string | null;
    min_confidence: number;
    motion_threshold_mps: number;
    track_gate_m: number;
    track_max_misses: number;
  };
  map: { bands: Band[]; uniform_cell_m: number };
  stream: {
    target_hz: number;
    playback_speed: number;
    max_stream_points: number;
    history_frames: number;
    loop_file_sequence: boolean;
  };
}

export interface ScheduleLevel {
  level: number;
  lower_m: number;
  upper_m: number;
  cell_size_m: number;
  subdiv: number;
}

export interface Schedule {
  quantum_m: number;
  tile_m: number;
  max_range_m: number;
  min_range_m: number;
  alignment: string;
  levels: ScheduleLevel[];
}

export interface Metrics {
  latency_ms: number;
  total_ms: number;
  capacity_fps: number;
  fps: number;
  sensor_sim_ms: number;
  points_in: number;
  points: number;
  occupied_cells: number;
  uniform_sparse_cells: number;
  uniform_dense_cells: number;
  reduction_vs_dense_pct: number;
  reduction_vs_sparse_pct: number;
  memory_proxy_mb: number;
  uniform_sparse_memory_proxy_mb: number;
  uniform_dense_memory_proxy_mb: number;
  measured_cell_array_mb: number;
  points_per_cell: number;
  class_confidence: number;
  objects: number;
  dynamic_objects: number;
  tracks: number;
  adaptive_projection_ms: number;
  uniform_projection_ms: number;
  partition_ok: boolean;
  seg_accuracy: number | null;
  seg_miou: number | null;
  rss_mb: number | null;
}

export interface BandProfile {
  level: number;
  label: string;
  cell_size_m: number;
  adaptive_cells: number;
  uniform_sparse_cells: number;
  points: number;
  adaptive_dense_cells: number;
}

export interface Comparison {
  adaptive_cells: number;
  uniform_sparse_cells: number;
  uniform_dense_cells: number;
  adaptive_dense_cells: number;
  reduction_vs_dense_pct: number;
  reduction_vs_sparse_pct: number;
  adaptive_memory_proxy_mb: number;
  uniform_sparse_memory_proxy_mb: number;
  uniform_dense_memory_proxy_mb: number;
  adaptive_projection_ms: number;
  uniform_projection_ms: number;
  band_profile: BandProfile[];
}

export interface SceneObject {
  id: number;
  track_id: number | null;
  class: number;
  x: number;
  y: number;
  length: number;
  width: number;
  yaw: number;
  height: number;
  min_z: number;
  max_z: number;
  distance: number;
  points: number;
  confidence: number;
  vx: number;
  vy: number;
  speed: number;
  dynamic: boolean;
  dynamic_score: number;
  track_hits: number;
  trail: [number, number][];
}

export interface EngineStatus {
  running: boolean;
  frame_index: number;
  latest_frame_id: number | null;
  history: number[];
  source: string;
  preset: string;
  sequence_length: number | null;
  segmenter: { name: string; kind: string; description: string };
  config_version: number;
  error: string | null;
  subscribers: number;
  uptime_s: number;
  rss_mb: number | null;
}

export interface FrameHeader {
  frame_id: number;
  sensor_time: number;
  timestamp: number;
  source: string;
  sensor: { x: number; y: number; z: number };
  metrics: Metrics;
  stage_ms: Record<string, number>;
  comparison: Comparison;
  partition: { ok: boolean; tiles: number; mixed_level_tiles: number; misassigned_cells: number };
  class_stats: { id: number; points: number; cells: number; mean_confidence: number }[];
  segmenter: { name: string; kind: string; description: string };
  segmentation_eval: { accuracy: number; miou: number; iou: (number | null)[]; confusion: number[][] } | null;
  preprocess: Record<string, number>;
  objects: SceneObject[];
  schedule: Schedule;
  counts: { cells: number; points: number; uniform: number };
  status: EngineStatus;
}

export interface CellArrays {
  level: Uint8Array;
  label: Uint8Array;
  ix: Int16Array;
  iy: Int16Array;
  count: Uint16Array;
  zmin: Int16Array;
  zmax: Int16Array;
  zmean: Int16Array;
  rough: Uint16Array;
  conf: Uint8Array;
  dyn: Uint8Array;
  occ: Uint8Array;
}

export interface PointArrays {
  x: Int16Array;
  y: Int16Array;
  z: Int16Array;
  label: Uint8Array;
  intensity: Uint8Array;
}

export interface UniformArrays {
  ix: Int16Array;
  iy: Int16Array;
  label: Uint8Array;
  zmax: Int16Array;
}

export interface Frame {
  header: FrameHeader;
  cells: CellArrays;
  points: PointArrays | null;
  uniform: UniformArrays | null;
  bytes: number;
  receivedAt: number;
}

export interface CellDetail {
  frame_id: number;
  level: number;
  cell_size: number;
  ix: number;
  iy: number;
  center_x: number;
  center_y: number;
  range_m: number;
  bounds: number[];
  point_count: number;
  min_z: number;
  max_z: number;
  mean_z: number;
  roughness: number;
  dominant_label: number;
  class_histogram: Record<string, number>;
  occupancy: number;
  dynamic_score: number;
  confidence: number;
  last_seen_frame: number;
  band: { level: number; lower_m: number; upper_m: number; cell_size_m: number };
}

/** A cell picked on the map, decoded from the stream (histogram is fetched separately). */
export interface PickedCell {
  mode: "adaptive" | "uniform";
  index: number;
  level: number;
  cellSize: number;
  ix: number;
  iy: number;
  cx: number;
  cy: number;
  range: number;
  count?: number;
  zmin?: number;
  zmax: number;
  zmean?: number;
  rough?: number;
  label: number;
  conf?: number;
  dyn?: number;
  occ?: number;
  frameId: number;
}

export type ColorMode = "class" | "elevation" | "level" | "confidence" | "dynamic";
export type ViewTab = "map" | "split" | "relief" | "points" | "architecture";

export interface DisplayOptions {
  colorMode: ColorMode;
  showBands: boolean;
  showGridDebug: boolean;
  showPoints: boolean;
  showObjects: boolean;
  showTracks: boolean;
  baseline: "adaptive" | "uniform";
  classVisible: boolean[];
}
