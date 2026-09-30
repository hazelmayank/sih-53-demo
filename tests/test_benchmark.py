"""Adaptive-vs-uniform comparison is computed from real data, on the same points."""

import numpy as np
import pytest

from backend.core.config import AppConfig, SceneConfig
from backend.core.pipeline import PerceptionPipeline
from backend.lidar.synthetic_scene import SyntheticLidar
from backend.mapping.baseline import compare, reduction_pct
from backend.mapping.cell import CELL_RECORD_BYTES, SPARSE_INDEX_BYTES_PER_CELL
from backend.mapping.projection import aggregate_cells


def test_reduction_formula():
    assert reduction_pct(100, 1000) == pytest.approx(90.0)
    assert reduction_pct(5, 0) == 0.0


def test_compare_uses_same_points(schedule, uniform):
    rng = np.random.default_rng(20)
    n = 30000
    r = np.sqrt(rng.uniform(0, 100 ** 2, n))
    a = rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.cos(a), r * np.sin(a)
    z = rng.normal(0, 0.1, n)
    lab = rng.integers(0, 6, n)
    ad = aggregate_cells(schedule, x, y, z, lab)
    un = aggregate_cells(uniform, x, y, z, lab)
    cmp = compare(schedule, uniform, ad, un)
    assert int(ad.point_count.sum()) == int(un.point_count.sum()) == n
    assert cmp.adaptive_cells == len(ad) and cmp.uniform_sparse_cells == len(un)
    assert cmp.adaptive_cells <= cmp.uniform_sparse_cells  # coarser far cells can only merge points
    assert cmp.uniform_dense_cells == sum(uniform.dense_cell_counts())
    assert cmp.reduction_vs_dense_pct == pytest.approx(100 * (1 - len(ad) / cmp.uniform_dense_cells))
    rec = CELL_RECORD_BYTES + SPARSE_INDEX_BYTES_PER_CELL
    assert cmp.adaptive_memory_proxy_mb == pytest.approx(len(ad) * rec / 2 ** 20)
    assert sum(b["adaptive_cells"] for b in cmp.band_profile) == len(ad)
    assert sum(b["uniform_sparse_cells"] for b in cmp.band_profile) == len(un)


def test_pipeline_metrics_are_measured():
    cfg = AppConfig(scene=SceneConfig(preset="road", beams=32))
    pipe = PerceptionPipeline(cfg)
    src = SyntheticLidar(cfg.scene)
    m = pipe.process(src.frame(0)).metrics
    assert m["latency_ms"] > 0 and m["adaptive_projection_ms"] > 0 and m["uniform_projection_ms"] > 0
    assert m["occupied_cells"] < m["uniform_sparse_cells"] < m["uniform_dense_cells"]
    assert 99.0 < m["reduction_vs_dense_pct"] < 100.0
    assert m["partition_ok"] is True
