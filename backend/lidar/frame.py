"""Internal point-cloud frame format shared by the synthetic generator and file importers.

Per point: x, y, z (m, sensor frame: x forward, y left, z up), intensity (0..1),
timestamp (s). Optional ground truth (synthetic data only): semantic label and
instance id. Ground truth is used for *evaluation only*, never by the
fallback classifier (except when the explicitly-labelled ``oracle_gt`` mode is
selected).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PointFrame:
    xyz: np.ndarray                  # (N, 3) float32
    intensity: np.ndarray            # (N,) float32 in [0, 1]
    timestamp: np.ndarray            # (N,) float64 seconds
    frame_index: int
    sensor_time: float
    source: str = "synthetic"
    gt_label: np.ndarray | None = None     # (N,) int8
    gt_instance: np.ndarray | None = None  # (N,) int32, -1 = terrain / none
    gt_objects: list[dict] = field(default_factory=list)
    sim_ms: float = 0.0

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def subset(self, mask: np.ndarray) -> "PointFrame":
        return PointFrame(
            xyz=self.xyz[mask],
            intensity=self.intensity[mask],
            timestamp=self.timestamp[mask],
            frame_index=self.frame_index,
            sensor_time=self.sensor_time,
            source=self.source,
            gt_label=None if self.gt_label is None else self.gt_label[mask],
            gt_instance=None if self.gt_instance is None else self.gt_instance[mask],
            gt_objects=self.gt_objects,
            sim_ms=self.sim_ms,
        )


def validate_points(xyz, intensity=None, timestamp=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coerce raw arrays into the internal format; raise ValueError on malformed data."""
    try:
        xyz = np.asarray(xyz, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed point data: {exc}") from exc
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"malformed point data: expected (N, 3+) array, got shape {xyz.shape}")
    xyz = np.ascontiguousarray(xyz[:, :3])
    n = xyz.shape[0]
    if intensity is None:
        intensity = np.zeros(n, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32).reshape(-1)
    if intensity.shape[0] != n:
        raise ValueError("malformed point data: intensity length does not match point count")
    if intensity.size and np.nanmax(intensity) > 1.5:  # 0..255 style reflectivity -> 0..1
        intensity = intensity / 255.0
    if timestamp is None:
        timestamp = np.zeros(n, dtype=np.float64)
    timestamp = np.asarray(timestamp, dtype=np.float64).reshape(-1)
    if timestamp.shape[0] != n:
        raise ValueError("malformed point data: timestamp length does not match point count")
    return xyz, np.clip(np.nan_to_num(intensity), 0.0, 1.0), timestamp
