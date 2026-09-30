"""Offline adaptive-vs-uniform benchmark.

Runs its own pipeline and frame source (the live stream is not disturbed) and
records, for every frame, the measured latency of both representations built
from the *same* points and the *same* semantic labels. A second pass sweeps the
sensor density to show how latency and cell counts scale with point count.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from backend.core.config import AppConfig
from backend.core.pipeline import PerceptionPipeline
from backend.lidar.synthetic_scene import SyntheticLidar

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"


def _stats(v) -> dict:
    a = np.asarray(v, dtype=np.float64)
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean()),
            "min": float(a.min()), "max": float(a.max())}


def run_benchmark(
    cfg: AppConfig,
    frames: int = 100,
    scaling_beams: tuple[int, ...] = (16, 32, 64, 96, 128),
    scaling_frames: int = 6,
    progress: Callable[[float, str], None] | None = None,
    save: bool = True,
) -> dict:
    cfg = cfg.model_copy(deep=True)
    cfg.scene.source = "synthetic"
    total = frames + scaling_frames * len(scaling_beams)
    done = 0

    def tick(msg: str) -> None:
        if progress:
            progress(done / max(total, 1), msg)

    pipe = PerceptionPipeline(cfg)
    src = SyntheticLidar(cfg.scene)
    rows = []
    t_run = time.perf_counter()
    for i in range(frames):
        pf = pipe.process(src.frame(i))
        m = pf.metrics
        rows.append({
            "frame": i, "points": m["points"], "latency_ms": m["latency_ms"],
            "adaptive_projection_ms": m["adaptive_projection_ms"], "uniform_projection_ms": m["uniform_projection_ms"],
            "adaptive_cells": m["occupied_cells"], "uniform_sparse_cells": m["uniform_sparse_cells"],
            "uniform_dense_cells": m["uniform_dense_cells"],
            "adaptive_mb": m["memory_proxy_mb"], "uniform_sparse_mb": m["uniform_sparse_memory_proxy_mb"],
            "uniform_dense_mb": m["uniform_dense_memory_proxy_mb"],
            "reduction_vs_dense_pct": m["reduction_vs_dense_pct"], "reduction_vs_sparse_pct": m["reduction_vs_sparse_pct"],
            "partition_ok": m["partition_ok"], "seg_accuracy": m["seg_accuracy"],
        })
        done += 1
        if i % 5 == 0:
            tick(f"frame {i + 1}/{frames}")

    scaling = []
    for beams in scaling_beams:
        c2 = cfg.model_copy(deep=True)
        c2.scene.beams = beams
        p2 = PerceptionPipeline(c2)
        s2 = SyntheticLidar(c2.scene)
        lat, ap, up, pts, ac, uc = [], [], [], [], [], []
        for i in range(scaling_frames):
            m = p2.process(s2.frame(i)).metrics
            lat.append(m["latency_ms"])
            ap.append(m["adaptive_projection_ms"])
            up.append(m["uniform_projection_ms"])
            pts.append(m["points"])
            ac.append(m["occupied_cells"])
            uc.append(m["uniform_sparse_cells"])
            done += 1
        tick(f"density sweep: {beams} beams")
        scaling.append({"beams": beams, "points": float(np.median(pts)), "latency_ms": float(np.median(lat[1:] or lat)),
                        "adaptive_projection_ms": float(np.median(ap[1:] or ap)),
                        "uniform_projection_ms": float(np.median(up[1:] or up)),
                        "adaptive_cells": float(np.median(ac)), "uniform_sparse_cells": float(np.median(uc))})

    warm = rows[2:] if len(rows) > 4 else rows
    result = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": time.perf_counter() - t_run,
        "config": {"preset": cfg.scene.preset, "seed": cfg.scene.seed, "beams": cfg.scene.beams,
                   "azimuth_resolution_deg": cfg.scene.azimuth_resolution_deg,
                   "bands": [b.model_dump() for b in cfg.map.bands], "uniform_cell_m": cfg.map.uniform_cell_m,
                   "max_range_m": cfg.processing.max_range_m, "segmenter": cfg.processing.segmenter},
        "frames": len(rows),
        "summary": {
            "points": _stats([r["points"] for r in warm]),
            "latency_ms": _stats([r["latency_ms"] for r in warm]),
            "adaptive_projection_ms": _stats([r["adaptive_projection_ms"] for r in warm]),
            "uniform_projection_ms": _stats([r["uniform_projection_ms"] for r in warm]),
            "adaptive_cells": _stats([r["adaptive_cells"] for r in warm]),
            "uniform_sparse_cells": _stats([r["uniform_sparse_cells"] for r in warm]),
            "uniform_dense_cells": float(warm[0]["uniform_dense_cells"]),
            "adaptive_mb": _stats([r["adaptive_mb"] for r in warm]),
            "uniform_sparse_mb": _stats([r["uniform_sparse_mb"] for r in warm]),
            "uniform_dense_mb": float(warm[0]["uniform_dense_mb"]),
            "reduction_vs_dense_pct": _stats([r["reduction_vs_dense_pct"] for r in warm]),
            "reduction_vs_sparse_pct": _stats([r["reduction_vs_sparse_pct"] for r in warm]),
            "partition_ok_all": all(r["partition_ok"] for r in rows),
            "seg_accuracy": _stats([r["seg_accuracy"] for r in warm if r["seg_accuracy"] is not None])
            if any(r["seg_accuracy"] is not None for r in warm) else None,
        },
        "per_frame": rows,
        "scaling": scaling,
    }
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(result, indent=1))
        result["saved_to"] = str(path.relative_to(RESULTS_DIR.parents[1]))
    if progress:
        progress(1.0, "done")
    return result
