"""Rolling runtime metrics (all values come from measured frames, never hard-coded)."""

from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

try:
    import psutil

    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover - psutil optional
    _PROC = None


def process_rss_mb() -> float | None:
    """Resident set size of the backend process (OS measurement, whole process)."""
    if _PROC is None:
        return None
    try:
        return _PROC.memory_info().rss / 2**20
    except Exception:  # pragma: no cover
        return None


class FpsMeter:
    """Measured output rate over a sliding wall-clock window."""

    def __init__(self, window_s: float = 3.0):
        self.window = window_s
        self.stamps: deque[float] = deque(maxlen=256)

    def tick(self) -> float:
        now = time.perf_counter()
        self.stamps.append(now)
        while len(self.stamps) > 2 and now - self.stamps[0] > self.window:
            self.stamps.popleft()
        return self.value()

    def value(self) -> float:
        if len(self.stamps) < 2:
            return 0.0
        dt = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / dt if dt > 0 else 0.0

    def reset(self) -> None:
        self.stamps.clear()


class MetricsHistory:
    KEYS = ("latency_ms", "total_ms", "fps", "points", "occupied_cells", "uniform_sparse_cells",
            "uniform_dense_cells", "reduction_vs_dense_pct", "reduction_vs_sparse_pct", "memory_proxy_mb",
            "uniform_sparse_memory_proxy_mb", "adaptive_projection_ms", "uniform_projection_ms",
            "dynamic_objects", "class_confidence", "seg_accuracy", "seg_miou")

    def __init__(self, maxlen: int = 600):
        self.rows: deque[dict] = deque(maxlen=maxlen)

    def add(self, frame_id: int, metrics: dict) -> None:
        self.rows.append({"frame_id": frame_id, **{k: metrics.get(k) for k in self.KEYS}})

    def clear(self) -> None:
        self.rows.clear()

    def summary(self) -> dict:
        if not self.rows:
            return {"frames": 0}
        out: dict = {"frames": len(self.rows)}
        for k in self.KEYS:
            vals = np.array([r[k] for r in self.rows if r.get(k) is not None], dtype=np.float64)
            if vals.size:
                out[k] = {"median": float(np.median(vals)), "p95": float(np.percentile(vals, 95)),
                          "min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean())}
        return out

    def series(self, last: int = 200) -> list[dict]:
        return list(self.rows)[-last:]
