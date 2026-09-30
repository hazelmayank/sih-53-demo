"""Class-agnostic obstacle clustering (connected components on a 2D occupancy grid)
and per-cluster geometric features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class ClusterFeatures:
    n: int
    count: np.ndarray       # points per cluster
    cx: np.ndarray          # oriented bounding-box centre
    cy: np.ndarray
    length: np.ndarray      # extent along principal axis
    width: np.ndarray       # extent along minor axis
    yaw: np.ndarray
    min_h: np.ndarray       # lowest / highest point above ground
    max_h: np.ndarray
    min_z: np.ndarray
    max_z: np.ndarray
    range_m: np.ndarray
    low_extent: np.ndarray  # principal-axis extent of the "body" band (0.2 m < h < 1.4 m)


def cluster_obstacles(
    xyz: np.ndarray,
    h: np.ndarray,
    obstacle_mask: np.ndarray,
    cell: float,
    extent: float,
    split_h: float = 2.0,
    join_from_h: float = 1.7,
    absorb_below_h: float = 0.8,
) -> tuple[np.ndarray, int, np.ndarray]:
    """2.5D connected-component clustering.

    Obstacle points are split into a *body* layer (h < split_h) and an *overhead*
    layer (h >= split_h), each labelled on a 2D occupancy grid. Components are
    joined only where a structure is vertically continuous across the split
    (body points close to split_h under an overhead component: trunks, poles,
    posts, tall vehicles). A pedestrian under a tree canopy or a car under a sign
    gantry therefore stays a separate object.

    Ground-level returns that fall in a body-layer cell whose structure reaches
    the ground (lowest point below ``absorb_below_h``) are absorbed into that
    structure (wall bases, pedestrians' feet).

    Returns (cluster id per point or -1, number of clusters, absorbed-terrain mask).
    """
    ids = np.full(xyz.shape[0], -1, dtype=np.int64)
    absorbed = np.zeros(xyz.shape[0], dtype=bool)
    idx = np.flatnonzero(obstacle_mask)
    if idx.size == 0:
        return ids, 0, absorbed
    n = int(np.ceil(2 * extent / cell))

    def cells(sel):
        gx = np.clip(((xyz[sel, 0] + extent) / cell).astype(np.int64), 0, n - 1)
        gy = np.clip(((xyz[sel, 1] + extent) / cell).astype(np.int64), 0, n - 1)
        return gx * n + gy

    lin = cells(idx)
    ho = h[idx]
    low = ho < split_h
    r = np.hypot(xyz[idx, 0], xyz[idx, 1])

    def occupancy(sel: np.ndarray) -> np.ndarray:
        # Range-adaptive connectivity: returns spread out with distance (and along surfaces
        # seen at grazing angles), so far occupancy is dilated before labelling.
        occ = np.zeros(n * n, dtype=bool)
        occ[lin[sel & (r < 25.0)]] = True
        for lo, hi, rad in ((25.0, 50.0, 1), (50.0, np.inf, 2)):
            g = lin[sel & (r >= lo) & (r < hi)]
            if g.size:
                gx, gy = np.divmod(np.unique(g), n)
                for dx in range(-rad, rad + 1):  # stamp a (2 rad + 1)^2 neighbourhood per far cell
                    for dy in range(-rad, rad + 1):
                        occ[np.clip(gx + dx, 0, n - 1) * n + np.clip(gy + dy, 0, n - 1)] = True
        return occ.reshape(n, n)

    lab_low, n_low = ndimage.label(occupancy(low), structure=_EIGHT)
    lab_high, n_high = ndimage.label(occupancy(~low), structure=_EIGHT)
    lab_low = lab_low.ravel()
    lab_high = lab_high.ravel()

    band = low & (ho >= join_from_h)
    a = lab_low[lin[band]] - 1
    b = lab_high[lin[band]] - 1
    ok = b >= 0
    total = n_low + n_high
    graph = coo_matrix((np.ones(int(ok.sum())), (a[ok], n_low + b[ok])), shape=(total, total))
    n_comp, comp = connected_components(graph, directed=False)

    ids[idx[low]] = comp[lab_low[lin[low]] - 1]
    ids[idx[~low]] = comp[n_low + lab_high[lin[~low]] - 1]

    # absorb ground-level returns under structures that reach the ground
    minh = np.full(n * n, np.inf)
    np.minimum.at(minh, lin[low], ho[low])
    t = np.flatnonzero(~obstacle_mask)
    if t.size:
        lt = cells(t)
        lab = lab_low[lt]
        take = (lab > 0) & (minh[lt] < absorb_below_h)
        ids[t[take]] = comp[lab[take] - 1]
        absorbed[t[take]] = True
    return ids, int(n_comp), absorbed


def cluster_features(xyz: np.ndarray, h: np.ndarray, cid: np.ndarray, n: int) -> ClusterFeatures:
    m = cid >= 0
    c = cid[m]
    x = xyz[m, 0].astype(np.float64)
    y = xyz[m, 1].astype(np.float64)
    z = xyz[m, 2].astype(np.float64)
    hh = h[m].astype(np.float64)
    cnt = np.bincount(c, minlength=n).astype(np.float64)
    safe = np.maximum(cnt, 1)
    mx = np.bincount(c, x, n) / safe
    my = np.bincount(c, y, n) / safe
    dx, dy = x - mx[c], y - my[c]
    sxx = np.bincount(c, dx * dx, n) / safe
    syy = np.bincount(c, dy * dy, n) / safe
    sxy = np.bincount(c, dx * dy, n) / safe
    yaw = 0.5 * np.arctan2(2 * sxy, sxx - syy)  # principal axis of the 2x2 covariance
    ca, sa = np.cos(yaw), np.sin(yaw)
    u = dx * ca[c] + dy * sa[c]
    v = -dx * sa[c] + dy * ca[c]

    def seg_min(vals):
        out = np.full(n, np.inf)
        np.minimum.at(out, c, vals)
        return out

    def seg_max(vals):
        out = np.full(n, -np.inf)
        np.maximum.at(out, c, vals)
        return out

    umin, umax, vmin, vmax = seg_min(u), seg_max(u), seg_min(v), seg_max(v)
    body = (hh > 0.2) & (hh < 1.4)
    bmin = np.full(n, np.inf)
    bmax = np.full(n, -np.inf)
    np.minimum.at(bmin, c[body], u[body])
    np.maximum.at(bmax, c[body], u[body])
    bw_min = np.full(n, np.inf)
    bw_max = np.full(n, -np.inf)
    np.minimum.at(bw_min, c[body], v[body])
    np.maximum.at(bw_max, c[body], v[body])
    low_extent = np.where(np.isfinite(bmin), np.maximum(bmax - bmin, bw_max - bw_min), 0.0)
    uc, vc = (umin + umax) / 2, (vmin + vmax) / 2
    cx = mx + uc * ca - vc * sa
    cy = my + uc * sa + vc * ca
    return ClusterFeatures(
        n=n, count=cnt.astype(np.int64), cx=cx, cy=cy,
        length=np.maximum(umax - umin, 0), width=np.maximum(vmax - vmin, 0), yaw=yaw,
        min_h=seg_min(hh), max_h=seg_max(hh), min_z=seg_min(z), max_z=seg_max(z),
        range_m=np.hypot(cx, cy), low_extent=low_extent,
    )
