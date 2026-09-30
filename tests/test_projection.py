"""2.5D aggregation: vectorised engine == reference loop, elevation + semantic statistics."""

import numpy as np
import pytest

from backend.core.config import NUM_CLASSES, SemanticClass as C
from backend.mapping.cell import aggregate_reference
from backend.mapping.projection import aggregate_cells


def test_vectorised_matches_reference(schedule):
    rng = np.random.default_rng(10)
    n = 6000
    r = np.sqrt(rng.uniform(0, 100 ** 2, n))
    a = rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.cos(a), r * np.sin(a)
    # cluster some points so cells get several points
    x[: n // 2] = rng.normal(5, 0.2, n // 2)
    y[: n // 2] = rng.normal(0, 0.2, n // 2)
    z = rng.normal(0, 1, n)
    lab = rng.integers(0, NUM_CLASSES, n)
    conf = rng.uniform(0.2, 1, n)
    dyn = rng.uniform(0, 1, n)
    fast = aggregate_cells(schedule, x, y, z, lab, conf, dyn, frame_id=7)
    ref = aggregate_reference(schedule, x, y, z, lab, conf, dyn, frame_id=7)
    assert len(fast) == len(ref)
    for i in range(len(fast)):
        c = ref[(int(fast.level[i]), int(fast.ix[i]), int(fast.iy[i]))]
        assert fast.point_count[i] == c.point_count
        assert fast.min_z[i] == pytest.approx(c.min_z, abs=1e-5)
        assert fast.max_z[i] == pytest.approx(c.max_z, abs=1e-5)
        assert fast.mean_z[i] == pytest.approx(c.mean_z, abs=1e-5)
        assert fast.roughness[i] == pytest.approx(c.roughness, abs=1e-4)
        assert fast.dominant_label[i] == c.dominant_label
        assert fast.confidence[i] == pytest.approx(c.confidence, abs=1e-5)
        assert fast.dynamic_score[i] == pytest.approx(c.dynamic_score, abs=1e-6)
        assert {k: v for k, v in enumerate(fast.class_histogram[i]) if v} == c.class_histogram
    assert fast.last_seen_frame == 7


def test_no_point_is_duplicated(schedule):
    rng = np.random.default_rng(11)
    n = 50000
    x = rng.uniform(-120, 120, n)
    y = rng.uniform(-120, 120, n)
    cells = aggregate_cells(schedule, x, y, np.zeros(n), np.zeros(n, np.int64))
    accepted = int(cells.accepted.sum())
    assert int(cells.point_count.sum()) == accepted
    assert cells.point_cell.shape[0] == accepted
    assert int(cells.class_histogram.sum()) == accepted
    assert np.unique(cells.key).size == len(cells)


def test_elevation_aggregation_keeps_overhang(schedule):
    # ground at z=0 under an overhanging canopy at 3 m inside one 5 cm cell
    x = np.array([2.01, 2.02, 2.03, 2.04])
    y = np.array([1.01, 1.02, 1.03, 1.04])
    z = np.array([0.0, 0.02, 3.0, 3.1])
    lab = np.array([C.DRIVABLE_TERRAIN, C.DRIVABLE_TERRAIN, C.STATIC_OBSTACLE, C.STATIC_OBSTACLE])
    cells = aggregate_cells(schedule, x, y, z, lab)
    assert len(cells) == 1
    assert cells.min_z[0] == pytest.approx(0.0)
    assert cells.max_z[0] == pytest.approx(3.1)
    assert cells.mean_z[0] == pytest.approx(np.mean(z))
    assert cells.roughness[0] == pytest.approx(np.std(z), rel=1e-5)


def test_mixed_class_cell_keeps_histogram(schedule):
    # a coarse far cell (50 cm at ~80 m) receiving several classes
    x = np.full(10, 80.1)
    y = np.linspace(0.05, 0.45, 10)
    lab = np.array([C.VEHICLE] * 5 + [C.PEDESTRIAN] * 3 + [C.UNKNOWN] * 2)
    cells = aggregate_cells(schedule, x, y, np.ones(10), lab)
    assert len(cells) == 1 and cells.level[0] == 3
    h = cells.class_histogram[0]
    assert h[C.VEHICLE] == 5 and h[C.PEDESTRIAN] == 3 and h[C.UNKNOWN] == 2
    assert cells.dominant_label[0] == C.VEHICLE
    assert cells.confidence[0] == pytest.approx(0.5)  # purity 5/10 x mean point confidence 1.0


def test_unknown_class_is_representable(schedule):
    cells = aggregate_cells(schedule, np.array([3.0]), np.array([0.0]), np.array([1.0]), np.array([C.UNKNOWN]))
    assert cells.dominant_label[0] == C.UNKNOWN


def test_empty_input(schedule):
    cells = aggregate_cells(schedule, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, np.int64))
    assert len(cells) == 0 and cells.memory_proxy_bytes() == 0


def test_cell_lookup(schedule):
    cells = aggregate_cells(schedule, np.array([45.3]), np.array([-12.7]), np.array([0.5]), np.array([C.VEHICLE]))
    lv, ix, iy = int(cells.level[0]), int(cells.ix[0]), int(cells.iy[0])
    assert cells.find(lv, ix, iy) == 0
    assert cells.find(lv, ix + 1, iy) is None
    d = cells.cell(0).to_dict()
    assert d["dominant_label"] == C.VEHICLE and d["cell_size"] == pytest.approx(0.25)
