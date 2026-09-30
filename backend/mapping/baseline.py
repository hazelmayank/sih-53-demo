"""Uniform-resolution baseline and adaptive-vs-uniform comparison.

Three representations are compared on the *same* frame and the *same* labels:

* dense uniform   - a full 5 cm grid over the whole mapped disk (what a naive
                    fixed-resolution map must allocate). Counted exactly.
* sparse uniform  - a 5 cm hash map that allocates only occupied cells
                    (actually computed by running the aggregation engine).
* sparse adaptive - the foveated map (actually computed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.mapping.adaptive_grid import ResolutionSchedule
from backend.mapping.cell import CELL_RECORD_BYTES, SPARSE_INDEX_BYTES_PER_CELL
from backend.mapping.projection import CellArrays

MB = 1024 * 1024


@dataclass
class BaselineComparison:
    adaptive_cells: int
    uniform_sparse_cells: int
    uniform_dense_cells: int
    adaptive_dense_cells: int
    reduction_vs_dense_pct: float
    reduction_vs_sparse_pct: float
    adaptive_memory_proxy_mb: float
    uniform_sparse_memory_proxy_mb: float
    uniform_dense_memory_proxy_mb: float
    adaptive_projection_ms: float
    uniform_projection_ms: float
    band_profile: list[dict]

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def reduction_pct(adaptive: int, uniform: int) -> float:
    if uniform <= 0:
        return 0.0
    return 100.0 * (1.0 - adaptive / uniform)


def compare(
    adaptive_schedule: ResolutionSchedule,
    uniform_schedule: ResolutionSchedule,
    adaptive: CellArrays,
    uniform: CellArrays,
) -> BaselineComparison:
    dense_uniform = sum(uniform_schedule.dense_cell_counts())
    dense_adaptive_per_level = adaptive_schedule.dense_cell_counts()

    # Range profile: bucket both maps into the adaptive range bands by cell-centre range.
    L = adaptive_schedule.num_levels
    a_cells = np.bincount(adaptive.level, minlength=L)
    a_points = np.bincount(adaptive.level, weights=adaptive.point_count, minlength=L)
    if len(uniform):
        u_r = np.hypot(uniform.center_x, uniform.center_y)
        u_band = adaptive_schedule.level_for_range(u_r)
        u_cells = np.bincount(u_band, minlength=L)
    else:
        u_cells = np.zeros(L, dtype=np.int64)
    profile = []
    for b in adaptive_schedule.bands:
        profile.append({
            "level": b.level,
            "label": f"{b.lower_m:g}-{b.upper_m:g} m",
            "cell_size_m": b.cell_size_m,
            "adaptive_cells": int(a_cells[b.level]),
            "uniform_sparse_cells": int(u_cells[b.level]),
            "points": int(a_points[b.level]),
            "adaptive_dense_cells": int(dense_adaptive_per_level[b.level]),
        })

    rec = CELL_RECORD_BYTES + SPARSE_INDEX_BYTES_PER_CELL
    return BaselineComparison(
        adaptive_cells=len(adaptive),
        uniform_sparse_cells=len(uniform),
        uniform_dense_cells=int(dense_uniform),
        adaptive_dense_cells=int(sum(dense_adaptive_per_level)),
        reduction_vs_dense_pct=reduction_pct(len(adaptive), dense_uniform),
        reduction_vs_sparse_pct=reduction_pct(len(adaptive), len(uniform)),
        adaptive_memory_proxy_mb=len(adaptive) * rec / MB,
        uniform_sparse_memory_proxy_mb=len(uniform) * rec / MB,
        # a dense grid needs no key/index, just the packed record per cell
        uniform_dense_memory_proxy_mb=dense_uniform * CELL_RECORD_BYTES / MB,
        adaptive_projection_ms=adaptive.elapsed_ms,
        uniform_projection_ms=uniform.elapsed_ms,
        band_profile=profile,
    )
