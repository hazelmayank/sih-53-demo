"""End-to-end perception pipeline for one frame.

LiDAR frame -> preprocessing -> ground estimation -> obstacle clustering ->
semantic inference -> object building + tracking -> adaptive projection &
2.5D aggregation -> uniform baseline on the same points/labels -> metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from backend.core.config import NUM_CLASSES, AppConfig, SemanticClass as C, TERRAIN_CLASSES
from backend.lidar.clustering import cluster_features, cluster_obstacles
from backend.lidar.frame import PointFrame
from backend.lidar.preprocess import estimate_ground, preprocess
from backend.lidar.segmentation import SegmentationContext, build_segmenter, evaluate
from backend.mapping.adaptive_grid import ResolutionSchedule
from backend.mapping.baseline import BaselineComparison, compare
from backend.mapping.projection import CellArrays, aggregate_cells, check_partition
from backend.tracking.tracker import Detection, NearestNeighbourTracker

TRACKABLE_MAX_EXTENT = 15.0


@dataclass
class CompactCells:
    """Minimal per-cell arrays kept for streaming the uniform baseline."""

    ix: np.ndarray
    iy: np.ndarray
    label: np.ndarray
    max_z: np.ndarray

    def __len__(self) -> int:
        return int(self.ix.shape[0])


@dataclass
class ProcessedFrame:
    frame_id: int
    sensor_time: float
    wall_time: float
    source: str
    schedule: ResolutionSchedule
    uniform_schedule: ResolutionSchedule
    cells: CellArrays
    uniform: CompactCells
    comparison: BaselineComparison
    objects: list[dict]
    stage_ms: dict[str, float]
    metrics: dict
    partition: dict
    class_stats: list[dict]
    segmenter: dict
    segmentation_eval: dict | None
    preprocess: dict
    sensor_height: float = 1.8
    stream_points: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    def summary(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "sensor_time": self.sensor_time,
            "timestamp": self.wall_time,
            "source": self.source,
            "sensor": {"x": 0.0, "y": 0.0, "z": self.sensor_height},
            "metrics": self.metrics,
            "stage_ms": self.stage_ms,
            "comparison": self.comparison.to_dict(),
            "partition": self.partition,
            "class_stats": self.class_stats,
            "segmenter": self.segmenter,
            "segmentation_eval": self.segmentation_eval,
            "preprocess": self.preprocess,
            "objects": self.objects,
            "schedule": self.schedule.describe(),
        }


class PerceptionPipeline:
    def __init__(self, cfg: AppConfig):
        self.tracker = NearestNeighbourTracker()
        self.configure(cfg)

    def configure(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        p = cfg.processing
        self.schedule = ResolutionSchedule.from_config(cfg.map, p.max_range_m, p.min_range_m)
        self.uniform_schedule = ResolutionSchedule.uniform(cfg.map.uniform_cell_m, p.max_range_m, p.min_range_m)
        self.segmenter = build_segmenter(p)
        self.tracker.gate_m = p.track_gate_m
        self.tracker.motion_threshold = p.motion_threshold_mps
        self.tracker.max_misses = p.track_max_misses

    def reset(self) -> None:
        self.tracker.reset()

    def process(self, raw: PointFrame, update_tracker: bool = True) -> ProcessedFrame:
        cfg = self.cfg
        p = cfg.processing
        stage: dict[str, float] = {}
        t_start = time.perf_counter()

        def lap(name: str, t0: float) -> float:
            t1 = time.perf_counter()
            stage[name] = (t1 - t0) * 1e3
            return t1

        t = time.perf_counter()
        frame, pstats = preprocess(raw, p)
        t = lap("preprocess", t)

        ground = estimate_ground(frame.xyz, p)
        h = (frame.xyz[:, 2] - ground.height_at(frame.xyz[:, 0], frame.xyz[:, 1])).astype(np.float32)
        terrain_mask = h < (p.obstacle_height_m + ground.terrain_tolerance(frame.xyz[:, 0], frame.xyz[:, 1]))
        t = lap("ground", t)

        cid, n_clusters, absorbed = cluster_obstacles(frame.xyz, h, ~terrain_mask, p.cluster_cell_m, p.max_range_m + 1.0)
        terrain_mask &= ~absorbed
        feats = cluster_features(frame.xyz, h, cid, n_clusters)
        t = lap("clustering", t)

        ctx = SegmentationContext(height=h, ground=ground, cluster_id=cid, clusters=feats,
                                  terrain_mask=terrain_mask, cfg=p, sensor_height=cfg.scene.sensor_height_m)
        seg = self.segmenter.predict(frame, ctx)
        labels, conf = seg.labels, seg.confidence
        t = lap("segmentation", t)

        objects, point_dynamic = self._objects(frame, labels, conf, cid, feats, update_tracker)
        t = lap("tracking", t)

        x, y, z = frame.xyz[:, 0], frame.xyz[:, 1], frame.xyz[:, 2]
        cells = aggregate_cells(self.schedule, x, y, z, labels, conf, point_dynamic, raw.frame_index)
        t = lap("adaptive_map", t)
        latency_ms = (t - t_start) * 1e3  # frame in -> adaptive map ready

        uni = aggregate_cells(self.uniform_schedule, x, y, z, labels, conf, point_dynamic, raw.frame_index)
        t = lap("uniform_baseline", t)
        comparison = compare(self.schedule, self.uniform_schedule, cells, uni)
        partition = check_partition(self.schedule, cells)

        seg_eval = None
        if frame.gt_label is not None and self.segmenter.kind != "oracle":
            seg_eval = evaluate(labels, frame.gt_label)

        n = len(frame)
        pc = np.bincount(labels.astype(np.int64), minlength=NUM_CLASSES)
        cc = np.bincount(cells.dominant_label, minlength=NUM_CLASSES)
        conf_by = np.bincount(labels.astype(np.int64), conf, NUM_CLASSES) / np.maximum(pc, 1)
        class_stats = [{"id": k, "points": int(pc[k]), "cells": int(cc[k]), "mean_confidence": float(conf_by[k])}
                       for k in range(NUM_CLASSES)]

        stream = self._stream_points(frame, labels)
        uniform_compact = CompactCells(ix=uni.ix.astype(np.int16), iy=uni.iy.astype(np.int16),
                                       label=uni.dominant_label, max_z=uni.max_z)
        # drop per-point bookkeeping so history frames stay small
        cells.point_cell = np.zeros(0, np.int64)
        cells.accepted = np.zeros(0, bool)
        t = lap("metrics", t)

        dynamic_objects = sum(1 for o in objects if o["dynamic"])
        metrics = {
            "latency_ms": latency_ms,
            "total_ms": (t - t_start) * 1e3,
            "capacity_fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0,
            "fps": 0.0,  # filled by the stream engine (measured wall-clock rate)
            "sensor_sim_ms": raw.sim_ms,
            "points_in": len(raw),
            "points": n,
            "occupied_cells": len(cells),
            "uniform_sparse_cells": comparison.uniform_sparse_cells,
            "uniform_dense_cells": comparison.uniform_dense_cells,
            "reduction_vs_dense_pct": comparison.reduction_vs_dense_pct,
            "reduction_vs_sparse_pct": comparison.reduction_vs_sparse_pct,
            "memory_proxy_mb": comparison.adaptive_memory_proxy_mb,
            "uniform_sparse_memory_proxy_mb": comparison.uniform_sparse_memory_proxy_mb,
            "uniform_dense_memory_proxy_mb": comparison.uniform_dense_memory_proxy_mb,
            "measured_cell_array_mb": cells.measured_array_bytes() / 2**20,
            "points_per_cell": n / max(len(cells), 1),
            "class_confidence": float(conf.mean()) if n else 0.0,
            "objects": len(objects),
            "dynamic_objects": dynamic_objects,
            "tracks": len(self.tracker.tracks),
            "adaptive_projection_ms": cells.elapsed_ms,
            "uniform_projection_ms": uni.elapsed_ms,
            "partition_ok": partition["ok"],
            "seg_accuracy": seg_eval["accuracy"] if seg_eval else None,
            "seg_miou": seg_eval["miou"] if seg_eval else None,
        }
        return ProcessedFrame(
            frame_id=raw.frame_index, sensor_time=raw.sensor_time, wall_time=time.time(), source=raw.source,
            schedule=self.schedule, uniform_schedule=self.uniform_schedule, cells=cells, uniform=uniform_compact,
            comparison=comparison, objects=objects, stage_ms=stage, metrics=metrics, partition=partition,
            class_stats=class_stats, segmenter=self.segmenter.info(), segmentation_eval=seg_eval,
            preprocess=pstats.to_dict(), stream_points=stream, sensor_height=cfg.scene.sensor_height_m,
        )

    # ------------------------------------------------------------------ helpers
    def _objects(self, frame, labels, conf, cid, feats, update_tracker):
        n = len(frame)
        point_dynamic = np.zeros(n, dtype=np.float32)
        if feats.n == 0:
            if update_tracker:
                self.tracker.update([], frame.sensor_time)
            return [], point_dynamic
        m = cid >= 0
        c = cid[m]
        votes = np.bincount(c * NUM_CLASSES + labels[m].astype(np.int64),
                            minlength=feats.n * NUM_CLASSES).reshape(feats.n, NUM_CLASSES)
        votes[:, list(TERRAIN_CLASSES)] = 0
        cls = np.argmax(votes, axis=1)
        cls[votes.sum(1) == 0] = C.UNKNOWN
        mean_conf = np.bincount(c, conf[m], feats.n) / np.maximum(feats.count, 1)

        keep = (feats.count >= self.cfg.processing.min_cluster_points) & (votes.sum(1) > 0)
        idx = np.flatnonzero(keep)
        dets = [Detection(float(feats.cx[i]), float(feats.cy[i]), int(cls[i]), int(i)) for i in idx
                if max(feats.length[i], feats.width[i]) < TRACKABLE_MAX_EXTENT]
        assigned = self.tracker.update(dets, frame.sensor_time) if update_tracker else self.tracker.match(dets)

        cluster_dyn = np.zeros(feats.n, dtype=np.float32)
        objects = []
        for i in idx:
            tr = assigned.get(int(i))
            dyn = self.tracker.dynamic_score(tr) if tr else 0.0
            cluster_dyn[i] = dyn
            objects.append({
                "id": int(i),
                "track_id": tr.id if tr else None,
                "class": int(tr.cls if tr and tr.cls != C.UNKNOWN else cls[i]),
                "x": float(feats.cx[i]), "y": float(feats.cy[i]),
                "length": float(max(feats.length[i], 0.1)), "width": float(max(feats.width[i], 0.1)),
                "yaw": float(feats.yaw[i]),
                "height": float(feats.max_h[i]),
                "min_z": float(feats.min_z[i]), "max_z": float(feats.max_z[i]),
                "distance": float(feats.range_m[i]),
                "points": int(feats.count[i]),
                "confidence": float(mean_conf[i]),
                "vx": tr.vx if tr else 0.0, "vy": tr.vy if tr else 0.0,
                "speed": tr.speed if tr else 0.0,
                "dynamic": bool(tr and self.tracker.is_dynamic(tr)),
                "dynamic_score": dyn,
                "track_hits": tr.hits if tr else 0,
                "trail": list(tr.trail) if tr and cls[i] in (C.VEHICLE, C.PEDESTRIAN) else [],
            })
        point_dynamic[m] = cluster_dyn[c]
        objects.sort(key=lambda o: o["distance"])
        return objects, point_dynamic

    def _stream_points(self, frame: PointFrame, labels: np.ndarray) -> dict[str, np.ndarray]:
        n = len(frame)
        k = min(n, self.cfg.stream.max_stream_points)
        if k <= 0:
            idx = np.zeros(0, dtype=np.int64)
        elif k == n:
            idx = np.arange(n)
        else:
            idx = np.linspace(0, n - 1, k).astype(np.int64)
        xyz = np.clip(np.rint(frame.xyz[idx] * 100.0), -32767, 32767).astype(np.int16)
        return {
            "p_x": xyz[:, 0].copy(), "p_y": xyz[:, 1].copy(), "p_z": xyz[:, 2].copy(),
            "p_label": labels[idx].astype(np.uint8),
            "p_int": np.clip(frame.intensity[idx] * 255, 0, 255).astype(np.uint8),
        }
