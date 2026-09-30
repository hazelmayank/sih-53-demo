"""Adaptive variable-resolution grid: resolution schedule, alignment tiles and cell keys.

Alignment strategy (why there are never gaps or overlaps between bands)
-----------------------------------------------------------------------
The requested cell sizes (5, 10, 25, 50 cm) are not a power-of-two hierarchy
(25 cm is 2.5x 10 cm), so a plain quadtree cannot nest them. Instead:

1.  Every cell size must be an integer multiple of a common *quantum* ``g``
    (the GCD of all sizes, 5 cm for the reference bands).
2.  All sizes must divide a common *alignment tile* ``T`` (their LCM, 50 cm
    for the reference bands). Every level's grid is anchored at the sensor
    origin, so every tile is covered exactly by an integer number of cells
    of *any* level (10x10 at 5 cm, 5x5 at 10 cm, 2x2 at 25 cm, 1x1 at 50 cm).
3.  A whole tile is assigned to exactly one resolution level, chosen by the
    planar range of the *tile centre* with half-open intervals
    ``[lower, upper)``. Every point in the tile inherits that level.

Consequences: the band boundary is a 0.5 m "staircase" approximation of the
circle, and the allocated cells form an exact partition of the plane -- no
point is ever stored twice, no two cells overlap, and no gaps appear at a
resolution transition. Integer arithmetic on the quantised coordinate makes
the nesting exact (no floating-point drift at the boundaries).

Range acceptance uses the *point* range: ``min_range <= r <= max_range``
(the outer bound is inclusive so a point at exactly 100 m is kept).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.core.config import MapConfig

QUANTUM_RESOLUTION_M = 0.001  # cell sizes must be whole millimetres
MAX_TILE_M = 5.0              # guard against sizes whose LCM would make huge tiles
SNAP_DECIMALS = 9             # coordinates are snapped to 1 nm before flooring

# CellKey packing: level in the top byte, ix / iy as 24-bit offset integers.
_KEY_OFFSET = 1 << 23
_KEY_MASK = (1 << 24) - 1


def pack_cell_key(level, ix, iy):
    """Pack (level, ix, iy) into one int64. Level is part of the key, so equal
    (ix, iy) at different resolutions can never collide."""
    level = np.asarray(level, dtype=np.int64)
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    if np.any(np.abs(ix) >= _KEY_OFFSET) or np.any(np.abs(iy) >= _KEY_OFFSET):
        raise ValueError("cell index out of packable range")
    return (level << 48) | ((ix + _KEY_OFFSET) << 24) | (iy + _KEY_OFFSET)


def unpack_cell_key(key):
    key = np.asarray(key, dtype=np.int64)
    level = key >> 48
    ix = ((key >> 24) & _KEY_MASK) - _KEY_OFFSET
    iy = (key & _KEY_MASK) - _KEY_OFFSET
    return level, ix, iy


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


@dataclass(frozen=True)
class Band:
    level: int
    lower_m: float
    upper_m: float
    cell_size_m: float
    subdiv: int  # cells per tile edge


@dataclass
class Projection:
    """Result of projecting points onto the grid (arrays align with the accepted points)."""

    accepted: np.ndarray  # bool mask over the input points
    level: np.ndarray     # int8
    ix: np.ndarray        # int32
    iy: np.ndarray        # int32
    key: np.ndarray       # int64 packed CellKey


class ResolutionSchedule:
    """Piecewise range -> cell-size schedule with exact tile alignment."""

    def __init__(self, uppers: list[float], sizes: list[float], max_range_m: float, min_range_m: float = 0.0):
        if len(uppers) != len(sizes) or not uppers:
            raise ValueError("uppers and sizes must be non-empty and the same length")
        if any(b <= a for a, b in zip(uppers, uppers[1:])):
            raise ValueError("band upper bounds must be strictly increasing")
        if max_range_m <= 0:
            raise ValueError("max_range must be positive")

        sizes_mm = []
        for s in sizes:
            mm = round(s / QUANTUM_RESOLUTION_M)
            if mm <= 0 or abs(mm * QUANTUM_RESOLUTION_M - s) > 1e-9:
                raise ValueError(f"cell size {s} m must be a positive whole number of millimetres")
            sizes_mm.append(mm)

        g_mm = sizes_mm[0]
        t_mm = sizes_mm[0]
        for mm in sizes_mm[1:]:
            g_mm = gcd(g_mm, mm)
            t_mm = _lcm(t_mm, mm)
        if t_mm * QUANTUM_RESOLUTION_M > MAX_TILE_M + 1e-9:
            raise ValueError(
                f"cell sizes {sizes} need a {t_mm / 1000:.3f} m alignment tile (> {MAX_TILE_M} m); "
                "choose sizes that share a small common multiple, e.g. 0.05/0.10/0.25/0.50"
            )

        self.max_range_m = float(max_range_m)
        self.min_range_m = float(min_range_m)
        self.quantum_m = g_mm * QUANTUM_RESOLUTION_M
        self.tile_m = t_mm * QUANTUM_RESOLUTION_M
        self._tile_q = t_mm // g_mm
        self._size_q = np.array([mm // g_mm for mm in sizes_mm], dtype=np.int64)
        self.uppers = np.array(uppers, dtype=np.float64)
        self.sizes = np.array([mm * QUANTUM_RESOLUTION_M for mm in sizes_mm], dtype=np.float64)
        lowers = [0.0] + list(uppers[:-1])
        self.bands = [
            Band(level=i, lower_m=float(lowers[i]), upper_m=float(uppers[i]),
                 cell_size_m=float(self.sizes[i]), subdiv=int(t_mm // sizes_mm[i]))
            for i in range(len(uppers))
        ]

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_config(cls, map_cfg: "MapConfig", max_range_m: float, min_range_m: float = 0.0) -> "ResolutionSchedule":
        return cls([b.upper_m for b in map_cfg.bands], [b.cell_size_m for b in map_cfg.bands], max_range_m, min_range_m)

    @classmethod
    def uniform(cls, cell_size_m: float, max_range_m: float, min_range_m: float = 0.0) -> "ResolutionSchedule":
        return cls([max(max_range_m, 1.0)], [cell_size_m], max_range_m, min_range_m)

    @property
    def num_levels(self) -> int:
        return len(self.bands)

    # ------------------------------------------------------------------ core rules
    def level_for_range(self, r) -> np.ndarray:
        """Half-open band lookup: r in [lower_i, upper_i) -> level i.

        Ranges at or beyond the last upper bound clamp to the last (coarsest)
        level; acceptance against max_range is handled separately.
        """
        r = np.asarray(r, dtype=np.float64)
        lvl = np.searchsorted(self.uppers, r, side="right")
        return np.minimum(lvl, self.num_levels - 1).astype(np.int8)

    def cell_size_for_range(self, r) -> np.ndarray:
        return self.sizes[self.level_for_range(r)]

    def quantize(self, v) -> np.ndarray:
        """Integer quantum index floor(v / g), snapped to kill float representation error."""
        return np.floor(np.round(np.asarray(v, dtype=np.float64) / self.quantum_m, SNAP_DECIMALS)).astype(np.int64)

    def accept_mask(self, x, y) -> np.ndarray:
        r = np.hypot(x, y)
        return (r >= self.min_range_m) & (r <= self.max_range_m)

    def project(self, x, y) -> Projection:
        """Map points to deterministic CellKeys (level, ix, iy)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        accepted = self.accept_mask(x, y)
        qx = self.quantize(x[accepted])
        qy = self.quantize(y[accepted])
        tx = qx // self._tile_q
        ty = qy // self._tile_q
        tile_r = np.hypot((tx + 0.5) * self.tile_m, (ty + 0.5) * self.tile_m)
        level = self.level_for_range(tile_r)
        sq = self._size_q[level]
        ix = qx // sq
        iy = qy // sq
        key = pack_cell_key(level, ix, iy)
        return Projection(accepted=accepted, level=level, ix=ix.astype(np.int32), iy=iy.astype(np.int32), key=key)

    def tile_level(self, tx, ty) -> np.ndarray:
        """Level assigned to alignment tile (tx, ty)."""
        tx = np.asarray(tx, dtype=np.float64)
        ty = np.asarray(ty, dtype=np.float64)
        return self.level_for_range(np.hypot((tx + 0.5) * self.tile_m, (ty + 0.5) * self.tile_m))

    def cell_bounds(self, level, ix, iy):
        """Axis-aligned bounds (x0, y0, x1, y1) of cells."""
        s = self.sizes[np.asarray(level, dtype=np.int64)]
        ix = np.asarray(ix, dtype=np.float64)
        iy = np.asarray(iy, dtype=np.float64)
        return ix * s, iy * s, (ix + 1) * s, (iy + 1) * s

    def cell_center(self, level, ix, iy):
        s = self.sizes[np.asarray(level, dtype=np.int64)]
        return (np.asarray(ix) + 0.5) * s, (np.asarray(iy) + 0.5) * s

    # ------------------------------------------------------------------ accounting
    def dense_cell_counts(self) -> list[int]:
        """Exact number of cells per level if the whole disk (r <= max_range) were
        allocated densely, using the same tile-based level assignment."""
        # Counted column-by-column in closed form so a uniform 5 cm grid over a
        # 100 m disk (~12.6 M cells) never has to be materialised.
        T = self.tile_m
        R = self.max_range_m
        n = int(np.ceil(R / T)) + 1
        cx = (np.arange(-n, n, dtype=np.float64) + 0.5) * T

        def tiles_within(radius: float, strict: bool) -> int:
            # number of tile centres with hypot(cx, cy) < radius (or <= when not strict)
            w2 = radius * radius - cx * cx
            ok = w2 > 0 if strict else w2 >= 0
            w = np.sqrt(np.where(ok, w2, 0.0)) / T
            lo, hi = -w - 0.5, w - 0.5
            cnt = (np.ceil(hi) - np.floor(lo) - 1) if strict else (np.floor(hi) - np.ceil(lo) + 1)
            return int(np.sum(np.where(ok, np.maximum(cnt, 0), 0)))

        inside_total = tiles_within(R, strict=False)
        tiles = []
        below_prev = 0
        for i, b in enumerate(self.bands):
            last = i == self.num_levels - 1
            below = inside_total if (last or b.upper_m > R) else tiles_within(b.upper_m, strict=True)
            tiles.append(max(below - below_prev, 0))
            below_prev = max(below, below_prev)
        return [tiles[b.level] * b.subdiv * b.subdiv for b in self.bands]

    def describe(self) -> dict:
        return {
            "quantum_m": self.quantum_m,
            "tile_m": self.tile_m,
            "max_range_m": self.max_range_m,
            "min_range_m": self.min_range_m,
            "alignment": "tile-centre half-open [lower, upper)",
            "levels": [
                {"level": b.level, "lower_m": b.lower_m, "upper_m": b.upper_m,
                 "cell_size_m": b.cell_size_m, "subdiv": b.subdiv}
                for b in self.bands
            ],
        }
