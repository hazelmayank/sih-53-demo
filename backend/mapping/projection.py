"""Vectorised point -> cell projection and 2.5D aggregation.

The grid is a sparse hash map keyed by the packed CellKey: only cells that
receive at least one point are allocated. Aggregation is done with a single
stable sort on the keys followed by segmented reductions, which is the NumPy
equivalent of the per-point dictionary loop in ``cell.aggregate_reference``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from backend.core.config import NUM_CLASSES
from backend.mapping.adaptive_grid import ResolutionSchedule
from backend.mapping.cell import (
    CELL_RECORD_BYTES,
    SPARSE_INDEX_BYTES_PER_CELL,
    Cell,
    occupancy_from_count,
)


@dataclass
class CellArrays:
    """Structure-of-arrays storage for all allocated cells of one frame."""

    key: np.ndarray            # int64 packed CellKey (sorted, unique)
    level: np.ndarray          # uint8
    ix: np.ndarray             # int32
    iy: np.ndarray             # int32
    point_count: np.ndarray    # uint32
    min_z: np.ndarray          # float32
    max_z: np.ndarray
    mean_z: np.ndarray
    roughness: np.ndarray
    dominant_label: np.ndarray  # uint8
    class_histogram: np.ndarray  # (n, NUM_CLASSES) uint32
    occupancy: np.ndarray
    dynamic_score: np.ndarray
    confidence: np.ndarray
    last_seen_frame: int
    cell_sizes: np.ndarray     # float64 size per level (lookup table)
    point_cell: np.ndarray = field(repr=False)  # cell index of each accepted point
    accepted: np.ndarray = field(repr=False)    # mask over input points
    elapsed_ms: float = 0.0

    def __len__(self) -> int:
        return int(self.key.shape[0])

    @property
    def cell_size(self) -> np.ndarray:
        return self.cell_sizes[self.level]

    @property
    def center_x(self) -> np.ndarray:
        return (self.ix + 0.5) * self.cell_size

    @property
    def center_y(self) -> np.ndarray:
        return (self.iy + 0.5) * self.cell_size

    def memory_proxy_bytes(self) -> int:
        """Packed-record estimate: record bytes + sparse index bytes per cell."""
        return len(self) * (CELL_RECORD_BYTES + SPARSE_INDEX_BYTES_PER_CELL)

    def measured_array_bytes(self) -> int:
        """Actual bytes held by this frame's NumPy cell arrays (excludes point index)."""
        arrays = (self.key, self.level, self.ix, self.iy, self.point_count, self.min_z, self.max_z,
                  self.mean_z, self.roughness, self.dominant_label, self.class_histogram,
                  self.occupancy, self.dynamic_score, self.confidence)
        return int(sum(a.nbytes for a in arrays))

    def find(self, level: int, ix: int, iy: int) -> int | None:
        from backend.mapping.adaptive_grid import pack_cell_key

        k = int(pack_cell_key(level, ix, iy))
        i = int(np.searchsorted(self.key, k))
        return i if i < len(self) and int(self.key[i]) == k else None

    def cell(self, i: int) -> Cell:
        s = float(self.cell_sizes[self.level[i]])
        hist = {c: int(v) for c, v in enumerate(self.class_histogram[i]) if v > 0}
        return Cell(
            level=int(self.level[i]), cell_size=s, ix=int(self.ix[i]), iy=int(self.iy[i]),
            center_x=(int(self.ix[i]) + 0.5) * s, center_y=(int(self.iy[i]) + 0.5) * s,
            point_count=int(self.point_count[i]), min_z=float(self.min_z[i]), max_z=float(self.max_z[i]),
            mean_z=float(self.mean_z[i]), roughness=float(self.roughness[i]),
            dominant_label=int(self.dominant_label[i]), class_histogram=hist,
            occupancy=float(self.occupancy[i]), dynamic_score=float(self.dynamic_score[i]),
            confidence=float(self.confidence[i]), last_seen_frame=int(self.last_seen_frame),
        )


def _sort_key(schedule: ResolutionSchedule, proj) -> np.ndarray:
    """Order-preserving compact key: int32 linear index per level when it fits, else the int64 key.

    The linear index is monotonic in the packed key (level-major, then ix, then iy),
    so sorted order and cell grouping are identical either way.
    """
    half = np.ceil((schedule.max_range_m + schedule.tile_m) / schedule.sizes).astype(np.int64) + 1
    span = 2 * half + 1
    sizes = span * span
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    if int(sizes.sum()) >= 2**31 - 1:
        return proj.key
    lv = proj.level.astype(np.int64)
    lin = offsets[lv] + (proj.ix + half[lv]) * span[lv] + (proj.iy + half[lv])
    return lin.astype(np.int32)


def aggregate_cells(
    schedule: ResolutionSchedule,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray | None = None,
    dynamic: np.ndarray | None = None,
    frame_id: int = 0,
) -> CellArrays:
    t0 = time.perf_counter()
    n_in = len(x)
    confidence = np.ones(n_in, dtype=np.float32) if confidence is None else confidence
    dynamic = np.zeros(n_in, dtype=np.float32) if dynamic is None else dynamic

    proj = schedule.project(x, y)
    acc = proj.accepted
    zf = np.asarray(z, dtype=np.float64)[acc]
    lab = np.asarray(labels, dtype=np.int64)[acc]
    conf = np.asarray(confidence, dtype=np.float64)[acc]
    dyn = np.asarray(dynamic, dtype=np.float32)[acc]
    n = zf.shape[0]

    if n == 0:
        empty_f = np.zeros(0, dtype=np.float32)
        return CellArrays(
            key=np.zeros(0, np.int64), level=np.zeros(0, np.uint8), ix=np.zeros(0, np.int32),
            iy=np.zeros(0, np.int32), point_count=np.zeros(0, np.uint32), min_z=empty_f, max_z=empty_f,
            mean_z=empty_f, roughness=empty_f, dominant_label=np.zeros(0, np.uint8),
            class_histogram=np.zeros((0, NUM_CLASSES), np.uint32), occupancy=empty_f,
            dynamic_score=empty_f, confidence=empty_f, last_seen_frame=frame_id,
            cell_sizes=schedule.sizes, point_cell=np.zeros(0, np.int64), accepted=acc,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )

    # --- group points by key: one sort, then segment boundaries. Stability is not
    # needed (all points of a cell share level/ix/iy), so the faster quicksort is used,
    # on a compact int32 linear index when the index space allows it.
    order = np.argsort(_sort_key(schedule, proj), kind="quicksort")
    skey = proj.key[order]
    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(skey[1:], skey[:-1], out=change[1:])
    starts = np.flatnonzero(change)
    n_cells = starts.shape[0]
    cell_of_sorted = np.cumsum(change) - 1
    point_cell = np.empty(n, dtype=np.int64)
    point_cell[order] = cell_of_sorted

    counts = np.diff(np.r_[starts, n])
    zs = zf[order]
    min_z = np.minimum.reduceat(zs, starts)
    max_z = np.maximum.reduceat(zs, starts)
    sum_z = np.add.reduceat(zs, starts)
    mean_z = sum_z / counts
    # roughness = population std of z, computed around the cell mean for stability
    dz = zs - np.repeat(mean_z, counts)
    roughness = np.sqrt(np.add.reduceat(dz * dz, starts) / counts)

    hist = np.bincount(cell_of_sorted * NUM_CLASSES + lab[order],
                       minlength=n_cells * NUM_CLASSES).reshape(n_cells, NUM_CLASSES)
    dominant = np.argmax(hist, axis=1)  # ties -> lowest label, same as the reference
    purity = hist[np.arange(n_cells), dominant] / counts
    mean_conf = np.add.reduceat(conf[order], starts) / counts
    dyn_max = np.maximum.reduceat(dyn[order], starts)

    ukey = skey[starts]
    first = order[starts]
    elapsed = (time.perf_counter() - t0) * 1e3
    return CellArrays(
        key=ukey,
        level=proj.level[first].astype(np.uint8),
        ix=proj.ix[first],
        iy=proj.iy[first],
        point_count=counts.astype(np.uint32),
        min_z=min_z.astype(np.float32),
        max_z=max_z.astype(np.float32),
        mean_z=mean_z.astype(np.float32),
        roughness=roughness.astype(np.float32),
        dominant_label=dominant.astype(np.uint8),
        class_histogram=hist.astype(np.uint32),
        occupancy=occupancy_from_count(counts).astype(np.float32),
        dynamic_score=dyn_max.astype(np.float32),
        confidence=(purity * mean_conf).astype(np.float32),
        last_seen_frame=frame_id,
        cell_sizes=schedule.sizes,
        point_cell=point_cell,
        accepted=acc,
        elapsed_ms=elapsed,
    )


def check_partition(schedule: ResolutionSchedule, cells: CellArrays) -> dict:
    """Runtime alignment check.

    Verifies (1) every alignment tile that holds cells holds cells of exactly
    one level (no overlaps / double allocation across a band transition) and
    (2) every cell's level equals the level the schedule assigns to its tile.
    """
    if len(cells) == 0:
        return {"ok": True, "tiles": 0, "mixed_level_tiles": 0, "misassigned_cells": 0}
    subdiv = np.array([b.subdiv for b in schedule.bands], dtype=np.int64)[cells.level]
    tx = np.floor_divide(cells.ix.astype(np.int64), subdiv)
    ty = np.floor_divide(cells.iy.astype(np.int64), subdiv)
    tile_key = (tx + (1 << 20)) * (1 << 21) + (ty + (1 << 20))
    order = np.argsort(tile_key, kind="stable")
    tk = tile_key[order]
    lv = cells.level[order].astype(np.int64)
    starts = np.flatnonzero(np.r_[True, tk[1:] != tk[:-1]])
    lv_min = np.minimum.reduceat(lv, starts)
    lv_max = np.maximum.reduceat(lv, starts)
    mixed = int(np.count_nonzero(lv_min != lv_max))
    expected = schedule.tile_level(tx, ty)
    misassigned = int(np.count_nonzero(expected != cells.level))
    return {"ok": mixed == 0 and misassigned == 0, "tiles": int(starts.shape[0]),
            "mixed_level_tiles": mixed, "misassigned_cells": misassigned}
