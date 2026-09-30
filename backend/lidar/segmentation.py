"""Semantic segmentation interface and implementations.

Any model is plugged in by implementing :class:`SegmentationModel.predict`:
it receives the preprocessed frame plus geometric context and returns one
label in ``SemanticClass`` and one confidence in [0, 1] per point. The mapping
engine never depends on which model produced the labels.

Implementations
---------------
* ``GeometricFallbackSegmenter`` - deterministic, rule-based heuristics
  (height above ground, local roughness / slope / step, reflectivity bands,
  cluster extent). It is NOT a trained neural network and the UI says so.
* ``TorchScriptSegmenter`` - adapter for a trained PointNet++ / KPConv /
  sparse-convolution network exported with TorchScript.
* ``OracleSegmenter`` - copies synthetic ground truth (debug / upper bound).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from backend.core.config import NUM_CLASSES, ProcessingConfig, SemanticClass as C
from backend.lidar.clustering import ClusterFeatures
from backend.lidar.frame import PointFrame
from backend.lidar.preprocess import GroundModel


@dataclass
class SegmentationContext:
    height: np.ndarray            # height above estimated ground, per point
    ground: GroundModel
    cluster_id: np.ndarray        # -1 for terrain candidates
    clusters: ClusterFeatures
    terrain_mask: np.ndarray      # geometric terrain candidates
    cfg: ProcessingConfig
    sensor_height: float = 1.8    # LiDAR origin height above the frame's z = 0


@dataclass
class SegmentationResult:
    labels: np.ndarray       # (N,) int8 in SemanticClass
    confidence: np.ndarray   # (N,) float32 in [0, 1]


class SegmentationModel(ABC):
    name: str = "base"
    kind: str = "abstract"      # "fallback" | "neural" | "oracle"
    description: str = ""

    @abstractmethod
    def predict(self, frame: PointFrame, ctx: SegmentationContext) -> SegmentationResult: ...

    def info(self) -> dict:
        return {"name": self.name, "kind": self.kind, "description": self.description}


# --------------------------------------------------------------------------- fallback
class GeometricFallbackSegmenter(SegmentationModel):
    name = "geometric_fallback"
    kind = "fallback"
    description = ("Deterministic rule-based fallback (height above ground, roughness, slope, step, "
                   "reflectivity bands, cluster shape). Not a trained neural network.")

    TERRAIN_CELL = 0.5
    ROUGH_MAX = 0.03
    STEP_MAX = 0.10
    SLOPE_MAX = 0.25
    DEPRESSION = -0.08
    CONCRETE_BAND = (0.34, 0.50)    # sidewalk / gravel reflectivity
    VEGETATION_BAND = (0.50, 0.78)  # grass / scrub reflectivity (paint is > 0.8)

    def predict(self, frame: PointFrame, ctx: SegmentationContext) -> SegmentationResult:
        n = len(frame)
        labels = np.full(n, int(C.UNKNOWN), dtype=np.int8)
        conf = np.full(n, 0.3, dtype=np.float32)
        drivable_near = self._terrain(frame, ctx, labels, conf)
        self._obstacles(frame, ctx, labels, conf, drivable_near)
        low = conf < ctx.cfg.min_confidence
        labels[low] = C.UNKNOWN
        return SegmentationResult(labels, conf)

    def _grid(self, xy: np.ndarray, extent: float):
        n = int(np.ceil(2 * extent / self.TERRAIN_CELL))
        q = np.clip(((xy + extent) / self.TERRAIN_CELL).astype(np.int64), 0, n - 1)
        return q[:, 0] * n + q[:, 1], n

    def _terrain(self, frame, ctx, labels, conf):
        """Label terrain candidates; returns a (n x n) mask of cells within 1 m of drivable terrain."""
        extent = ctx.cfg.max_range_m + 1.0
        t = np.flatnonzero(ctx.terrain_mask)
        inv_all, n = self._grid(frame.xyz[t, :2], extent)
        if t.size == 0:
            return np.zeros((n, n), dtype=bool)
        xyz = frame.xyz[t]
        h = ctx.height[t].astype(np.float64)
        it = frame.intensity[t]
        # dense 0.5 m terrain-cell statistics (fixed grid, no sort needed)
        inv = inv_all
        k = n * n
        cnt = np.maximum(np.bincount(inv, minlength=k), 1)
        z = xyz[:, 2].astype(np.float64)
        mean_z = np.bincount(inv, z, k) / cnt
        rough = np.sqrt(np.maximum(np.bincount(inv, (z - mean_z[inv]) ** 2, k) / cnt, 0))
        zmin = np.full(k, np.inf)
        zmax = np.full(k, -np.inf)
        np.minimum.at(zmin, inv, z)
        np.maximum.at(zmax, inv, z)
        mean_h = np.bincount(inv, h, k) / cnt
        concrete = np.bincount(inv, ((it >= self.CONCRETE_BAND[0]) & (it < self.CONCRETE_BAND[1])).astype(float), k) / cnt
        veg = np.bincount(inv, ((it >= self.VEGETATION_BAND[0]) & (it < self.VEGETATION_BAND[1])).astype(float), k) / cnt
        slope = ctx.ground.slope_at(xyz[:, 0], xyz[:, 1])

        depress = mean_h[inv] < self.DEPRESSION
        steep = slope > self.SLOPE_MAX
        has_stats = cnt[inv] >= 3
        step = has_stats & ((zmax - zmin)[inv] > self.STEP_MAX)
        rough_p = has_stats & (rough[inv] > self.ROUGH_MAX)
        material = (concrete[inv] > 0.5) | (veg[inv] > 0.5)

        nd = depress | steep | step | rough_p | material
        labels[t] = np.where(nd, C.NON_DRIVABLE_TERRAIN, C.DRIVABLE_TERRAIN)
        support = np.minimum(1.0, 0.55 + 0.1 * cnt[inv])  # fewer returns -> less evidence
        c = np.where(nd, np.where(depress | steep | step, 0.8, 0.7), 0.88) * support
        conf[t] = c.astype(np.float32)

        drv = np.bincount(inv, (~nd).astype(float), k) / cnt > 0.5
        return ndimage.binary_dilation(drv.reshape(n, n), iterations=2)

    def _obstacles(self, frame, ctx, labels, conf, drivable_near):
        cf = ctx.clusters
        if cf.n == 0:
            return
        m = ctx.cluster_id >= 0
        cid = ctx.cluster_id[m]
        # fraction of each cluster's points standing over / next to drivable terrain
        g, n = self._grid(frame.xyz[m, :2], ctx.cfg.max_range_m + 1.0)
        on_road = np.bincount(cid, drivable_near.ravel()[g].astype(float), cf.n) / np.maximum(cf.count, 1)

        height = cf.max_h
        L = np.maximum(cf.length, cf.width)
        W = np.minimum(cf.length, cf.width)
        floating = cf.min_h > 1.4
        # Top clipped by the vertical field of view: the object continues above the highest
        # beam (poles / trees / walls close to the sensor). A pedestrian is shorter than the
        # sensor mount, so its top is always inside the FOV.
        r_all = np.hypot(frame.xyz[:, 0], frame.xyz[:, 1])
        elev = np.arctan2(frame.xyz[:, 2] - ctx.sensor_height, np.maximum(r_all, 0.1))
        top_elev = float(np.percentile(elev, 99.9)) if elev.size else 0.05
        d_near = np.maximum(cf.range_m - 0.5 * L, 0.5)
        clipped = cf.max_z >= ctx.sensor_height + d_near * np.tan(top_elev) - 0.2
        cls = np.full(cf.n, int(C.STATIC_OBSTACLE), dtype=np.int8)
        cc = np.full(cf.n, 0.6, dtype=np.float32)

        ped = (~floating) & ~clipped & (height > 1.1) & (height < 2.2) & (L < 1.1) & (W < 0.9)
        pole = (~floating) & (L < 1.0) & ((height >= 2.2) | (clipped & (height > 1.1)))
        wall = (L >= 13.0) | (height >= 4.2)
        vehicle = ((~floating) & ~ped & ~wall & (height > 1.0) & (L >= 1.4) & (W < 3.2)
                   & (cf.low_extent >= 1.0)        # solid body near the ground (rejects trees, shelters)
                   & (on_road >= 0.4))             # vehicles stand on drivable terrain

        cls[ped] = C.PEDESTRIAN
        cc[ped] = 0.72
        cls[vehicle] = C.VEHICLE
        cc[vehicle] = 0.55 + 0.3 * np.minimum(on_road[vehicle], 1.0)
        cc[pole | wall | floating] = 0.85
        few = cf.count < ctx.cfg.min_cluster_points
        cls[few] = C.UNKNOWN
        cc[few] = 0.25
        # sparse long-range clusters carry less shape evidence
        cc = cc * np.minimum(1.0, 0.5 + cf.count / 40.0).astype(np.float32)

        labels[m] = cls[cid]
        conf[m] = cc[cid]


# --------------------------------------------------------------------------- neural adapter
class TorchScriptSegmenter(SegmentationModel):
    """Adapter for a trained point-wise network (PointNet++ / KPConv / MinkowskiNet...).

    Contract for the exported model: ``forward(points: FloatTensor[B, N, 4]) ->
    logits: FloatTensor[B, N, K]`` where the 4 input channels are x, y, z,
    intensity in the sensor frame. ``class_map`` maps the model's K output
    classes onto the 6 map classes (e.g. SemanticKITTI road/parking ->
    DRIVABLE_TERRAIN, sidewalk/terrain/vegetation -> NON_DRIVABLE_TERRAIN,
    building/fence/pole/trunk -> STATIC_OBSTACLE, car/truck/bus -> VEHICLE,
    person/bicyclist -> PEDESTRIAN).
    """

    name = "torchscript"
    kind = "neural"

    def __init__(self, model_path: str, class_map: list[int] | None = None, chunk: int = 131072, device: str | None = None):
        try:
            import torch  # optional dependency
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed: pip install torch") from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        self.class_map = np.asarray(class_map if class_map is not None else list(range(NUM_CLASSES)), dtype=np.int8)
        self.chunk = chunk
        self.description = f"TorchScript model {model_path} on {self.device}"

    def predict(self, frame: PointFrame, ctx: SegmentationContext) -> SegmentationResult:
        torch = self._torch
        feats = np.hstack([frame.xyz, frame.intensity[:, None]]).astype(np.float32)
        labels = np.empty(len(frame), dtype=np.int8)
        conf = np.empty(len(frame), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(frame), self.chunk):
                x = torch.from_numpy(feats[s:s + self.chunk]).to(self.device)[None]
                prob = torch.softmax(self.model(x)[0], dim=-1)
                p, k = prob.max(dim=-1)
                labels[s:s + self.chunk] = self.class_map[k.cpu().numpy()]
                conf[s:s + self.chunk] = p.cpu().numpy()
        labels[conf < ctx.cfg.min_confidence] = C.UNKNOWN
        return SegmentationResult(labels, conf)


# --------------------------------------------------------------------------- oracle
class OracleSegmenter(SegmentationModel):
    name = "oracle_gt"
    kind = "oracle"
    description = "Synthetic ground-truth labels (debug / upper bound). Only available for synthetic scenes."

    def predict(self, frame: PointFrame, ctx: SegmentationContext) -> SegmentationResult:
        if frame.gt_label is None:
            raise RuntimeError("oracle_gt needs synthetic ground truth; use geometric_fallback for imported files")
        return SegmentationResult(frame.gt_label.astype(np.int8).copy(), np.ones(len(frame), dtype=np.float32))


def build_segmenter(cfg: ProcessingConfig) -> SegmentationModel:
    if cfg.segmenter == "torchscript":
        if not cfg.model_path:
            raise ValueError("segmenter=torchscript requires processing.model_path")
        return TorchScriptSegmenter(cfg.model_path)
    if cfg.segmenter == "oracle_gt":
        return OracleSegmenter()
    return GeometricFallbackSegmenter()


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Point-wise accuracy and per-class IoU against ground truth."""
    k = NUM_CLASSES
    cm = np.bincount(gt.astype(np.int64) * k + pred.astype(np.int64), minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1) - tp
    present = cm.sum(1) > 0
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "miou": float(np.nanmean(iou[present])) if present.any() else 0.0,
        "iou": [None if np.isnan(v) else float(v) for v in iou],
        "confusion": cm.tolist(),  # rows = ground truth, cols = prediction
    }
