"""2.5D cell data model.

A cell summarises an x-y footprint while keeping elevation (min/max/mean,
roughness) and semantics (full class histogram), so information is not
destroyed when distant cells become coarse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from backend.core.config import NUM_CLASSES

# Compact binary record used for the memory proxy: this is what one cell would
# cost in a packed C struct / numpy structured array (no Python object overhead).
CELL_RECORD_DTYPE = np.dtype([
    ("level", np.uint8),
    ("dominant_label", np.uint8),
    ("ix", np.int32),
    ("iy", np.int32),
    ("point_count", np.uint32),
    ("min_z", np.float32),
    ("max_z", np.float32),
    ("mean_z", np.float32),
    ("roughness", np.float32),
    ("class_histogram", np.uint16, (NUM_CLASSES,)),
    ("occupancy", np.float32),
    ("dynamic_score", np.float32),
    ("confidence", np.float32),
    ("last_seen_frame", np.uint32),
], align=False)

CELL_RECORD_BYTES = CELL_RECORD_DTYPE.itemsize
# Sparse hash-map overhead per allocated cell: 8-byte packed key + 8-byte slot/index.
SPARSE_INDEX_BYTES_PER_CELL = 16

OCCUPANCY_K = 2.0  # occupancy = 1 - exp(-count / K): evidence that the cell is really occupied


def occupancy_from_count(count):
    return 1.0 - np.exp(-np.asarray(count, dtype=np.float64) / OCCUPANCY_K)


@dataclass
class Cell:
    level: int
    cell_size: float
    ix: int
    iy: int
    center_x: float
    center_y: float
    point_count: int = 0
    min_z: float = math.inf
    max_z: float = -math.inf
    mean_z: float = 0.0
    roughness: float = 0.0
    dominant_label: int = 5
    class_histogram: dict[int, int] = field(default_factory=dict)
    occupancy: float = 0.0
    dynamic_score: float = 0.0
    confidence: float = 0.0
    last_seen_frame: int = 0
    # Welford accumulator for roughness (running variance of z)
    _m2: float = 0.0
    _conf_sum: float = 0.0

    @property
    def range_m(self) -> float:
        return math.hypot(self.center_x, self.center_y)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "cell_size": self.cell_size,
            "ix": self.ix,
            "iy": self.iy,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "range_m": self.range_m,
            "point_count": self.point_count,
            "min_z": self.min_z,
            "max_z": self.max_z,
            "mean_z": self.mean_z,
            "roughness": self.roughness,
            "dominant_label": self.dominant_label,
            "class_histogram": {str(k): v for k, v in sorted(self.class_histogram.items())},
            "occupancy": self.occupancy,
            "dynamic_score": self.dynamic_score,
            "confidence": self.confidence,
            "last_seen_frame": self.last_seen_frame,
        }


def aggregate_reference(schedule, x, y, z, labels, confidence=None, dynamic=None, frame_id: int = 0) -> dict:
    """Straightforward per-point implementation of the specification's pseudocode.

    Slow; used by tests to validate the vectorised engine in ``projection.py``.
    Returns ``{(level, ix, iy): Cell}``.
    """
    n = len(x)
    confidence = np.ones(n) if confidence is None else confidence
    dynamic = np.zeros(n) if dynamic is None else dynamic
    proj = schedule.project(x, y)
    idx = np.flatnonzero(proj.accepted)
    grid: dict[tuple[int, int, int], Cell] = {}
    for j, i in enumerate(idx):
        level, ix, iy = int(proj.level[j]), int(proj.ix[j]), int(proj.iy[j])
        key = (level, ix, iy)
        cell = grid.get(key)
        if cell is None:
            s = float(schedule.sizes[level])
            cell = Cell(level=level, cell_size=s, ix=ix, iy=iy,
                        center_x=(ix + 0.5) * s, center_y=(iy + 0.5) * s)
            grid[key] = cell
        zi = float(z[i])
        cell.point_count += 1
        cell.min_z = min(cell.min_z, zi)
        cell.max_z = max(cell.max_z, zi)
        delta = zi - cell.mean_z
        cell.mean_z += delta / cell.point_count            # running mean
        cell._m2 += delta * (zi - cell.mean_z)              # Welford update for roughness
        lbl = int(labels[i])
        cell.class_histogram[lbl] = cell.class_histogram.get(lbl, 0) + 1
        cell.dynamic_score = max(cell.dynamic_score, float(dynamic[i]))
        cell._conf_sum += float(confidence[i])
        cell.last_seen_frame = frame_id

    for cell in grid.values():
        cell.roughness = math.sqrt(max(cell._m2 / cell.point_count, 0.0))
        # argmax with lowest-label tie-break (matches the vectorised engine)
        cell.dominant_label = min(cell.class_histogram, key=lambda k: (-cell.class_histogram[k], k))
        purity = cell.class_histogram[cell.dominant_label] / cell.point_count
        cell.confidence = purity * (cell._conf_sum / cell.point_count)
        cell.occupancy = float(occupancy_from_count(cell.point_count))
    return grid
