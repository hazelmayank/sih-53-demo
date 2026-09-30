"""Preprocessing: sanitising, range gating, outlier removal, optional voxel downsampling,
and terrain (ground) height estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from backend.core.config import ProcessingConfig
from backend.lidar.frame import PointFrame


@dataclass
class PreprocessStats:
    input_points: int
    non_finite: int
    out_of_range: int
    outliers: int
    downsampled: int
    output_points: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _voxel_keys(xyz: np.ndarray, size: float) -> np.ndarray:
    q = np.floor(xyz / size).astype(np.int64) + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def preprocess(frame: PointFrame, cfg: ProcessingConfig) -> tuple[PointFrame, PreprocessStats]:
    n0 = len(frame)
    finite = np.isfinite(frame.xyz).all(axis=1) & np.isfinite(frame.intensity)
    f = frame.subset(finite) if not finite.all() else frame

    r = np.hypot(f.xyz[:, 0], f.xyz[:, 1])
    in_range = (r >= cfg.min_range_m) & (r <= cfg.max_range_m) & (f.xyz[:, 2] > -20) & (f.xyz[:, 2] < 40)
    f1 = f.subset(in_range)

    n_out = 0
    if cfg.outlier_filter and len(f1):
        # A return is an outlier when its 1 m voxel and all 26 neighbours hold no other point.
        # (Neighbourhood test keeps sparse-but-real long-range structure intact.)
        # Dense voxel counts (with a 1-voxel pad) avoid a sort over all points.
        q = np.floor(f1.xyz).astype(np.int64)
        lo = q.min(axis=0) - 1
        dims = q.max(axis=0) - lo + 2
        q -= lo
        lin = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
        counts = np.bincount(lin, minlength=int(dims.prod()))
        lonely = np.flatnonzero(counts[lin] == 1)
        drop = np.zeros(len(f1), dtype=bool)
        if lonely.size:
            base = lin[lonely]
            neigh = np.zeros(lonely.size, dtype=np.int64)
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    for ddz in (-1, 0, 1):
                        neigh += counts[base + (ddx * dims[1] + ddy) * dims[2] + ddz]
            drop[lonely[neigh == 1]] = True
            n_out = int(drop.sum())
            if n_out:
                f1 = f1.subset(~drop)

    n_ds = 0
    if cfg.voxel_downsample_m > 0 and len(f1):
        keys = _voxel_keys(f1.xyz, cfg.voxel_downsample_m)
        _, first = np.unique(keys, return_index=True)
        mask = np.zeros(len(f1), dtype=bool)
        mask[np.sort(first)] = True
        n_ds = len(f1) - int(mask.sum())
        f1 = f1.subset(mask)

    stats = PreprocessStats(n0, int((~finite).sum()), int((~in_range).sum()), n_out, n_ds, len(f1))
    return f1, stats


# --------------------------------------------------------------------------- ground model
def _range_bin_edges(max_range: float) -> np.ndarray:
    """Polar range bins: 0.5 m to 20 m, 1 m to 50 m, 2 m beyond (matches ring spacing growth)."""
    e = np.concatenate([np.arange(0.0, 20.0, 0.5), np.arange(20.0, 50.0, 1.0), np.arange(50.0, max_range + 2.0, 2.0)])
    return np.append(e[e < max_range + 1.0], max_range + 1.0)


@dataclass
class GroundModel:
    """Terrain height on a polar (azimuth sector x range bin) grid centred on the sensor."""

    z: np.ndarray        # (S, B) ground height per sector / range bin
    slope: np.ndarray    # (S, B) radial slope magnitude (rise / run)
    edges: np.ndarray    # (B + 1,) range-bin edges
    n_sectors: int

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def _sector(self, x, y) -> np.ndarray:
        a = np.arctan2(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64))
        return (np.floor((a + np.pi) / (2 * np.pi) * self.n_sectors).astype(np.int64)) % self.n_sectors

    def height_at(self, x, y) -> np.ndarray:
        """Ground height with linear interpolation along range inside the point's sector."""
        s = self._sector(x, y)
        r = np.hypot(x, y)
        c = self.centers
        j0 = np.clip(np.searchsorted(c, r) - 1, 0, c.shape[0] - 2)
        w = np.clip((r - c[j0]) / (c[j0 + 1] - c[j0]), 0.0, 1.0)
        B = c.shape[0]
        zf = self.z.ravel()
        k = s * B + j0
        return (1 - w) * zf[k] + w * zf[k + 1]

    def slope_at(self, x, y) -> np.ndarray:
        s = self._sector(x, y)
        b = np.clip(np.searchsorted(self.edges, np.hypot(x, y), side="right") - 1, 0, self.z.shape[1] - 1)
        return self.slope[s, b]

    def terrain_tolerance(self, x, y) -> np.ndarray:
        """Extra height allowed for terrain on slopes (half a bin of rise)."""
        b = np.clip(np.searchsorted(self.edges, np.hypot(x, y), side="right") - 1, 0, self.z.shape[1] - 1)
        width = np.diff(self.edges)[b]
        return 0.5 * self.slope_at(x, y) * width


def estimate_ground(xyz: np.ndarray, cfg: ProcessingConfig) -> GroundModel:
    """Polar line-tracking ground estimation (Himmelsbach-style).

    For every azimuth sector the lowest return of each range bin is visited
    outwards from the sensor. It is accepted as ground when it does not rise
    more than ``max_slope * dr + tol`` above the last accepted ground and the
    bin is not a vertical structure (large z spread starting above ground).
    Rejected / empty bins hold the last ground height, so object-only bins
    (car sides, walls) and occluded regions never lift the terrain estimate.
    """
    S = cfg.ground_sectors
    max_slope = cfg.ground_max_slope
    tol = 0.18
    edges = _range_bin_edges(cfg.max_range_m)
    B = edges.shape[0] - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)

    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    z = xyz[:, 2].astype(np.float64)
    r = np.hypot(x, y)
    s = (np.floor((np.arctan2(y, x) + np.pi) / (2 * np.pi) * S).astype(np.int64)) % S
    b = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, B - 1)
    lin = s * B + b
    zmin = np.full(S * B, np.inf)
    zmax = np.full(S * B, -np.inf)
    np.minimum.at(zmin, lin, z)
    np.maximum.at(zmax, lin, z)
    zmin = zmin.reshape(S, B)
    zmax = zmax.reshape(S, B)

    # initial ground: robust near-field level (median of the lowest returns within 8 m)
    near = zmin[:, centers < 8.0]
    near = near[np.isfinite(near)]
    g0 = float(np.median(near)) if near.size else (float(np.percentile(z, 5)) if z.size else 0.0)

    ground = np.empty((S, B))
    prev = np.full(S, g0)
    prev_r = np.zeros(S)
    for j in range(B):
        c = zmin[:, j]
        dr = centers[j] - prev_r
        rise = c - prev
        vertical = (zmax[:, j] - c > 0.5 + 0.4 * width[j]) & (rise > 0.1)
        ok = np.isfinite(c) & (rise <= max_slope * dr + tol) & ~vertical
        ground[:, j] = np.where(ok, c, prev)
        prev = np.where(ok, c, prev)
        prev_r = np.where(ok, centers[j], prev_r)

    # light smoothing across neighbouring sectors (circular)
    ground = ndimage.uniform_filter1d(ground, size=3, axis=0, mode="wrap")
    slope = np.abs(np.gradient(ground, centers, axis=1))
    return GroundModel(z=ground, slope=slope, edges=edges, n_sectors=S)
