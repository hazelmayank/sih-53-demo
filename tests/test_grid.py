"""Schedule construction, CellKey uniqueness and exact partition of the plane."""

import numpy as np
import pytest

from backend.mapping.adaptive_grid import ResolutionSchedule, pack_cell_key, unpack_cell_key
from backend.mapping.projection import aggregate_cells, check_partition


def test_reference_schedule_alignment(schedule):
    assert schedule.quantum_m == pytest.approx(0.05)
    assert schedule.tile_m == pytest.approx(0.5)
    assert [b.subdiv for b in schedule.bands] == [10, 5, 2, 1]


def test_invalid_schedules_rejected():
    with pytest.raises(ValueError):
        ResolutionSchedule([30, 10], [0.05, 0.1], 100)  # not increasing
    with pytest.raises(ValueError):
        ResolutionSchedule([10, 100], [0.37, 0.41], 100)  # alignment tile would be 151.7 m
    with pytest.raises(ValueError):
        ResolutionSchedule([10, 100], [0.0505, 0.1], 100)  # not whole millimetres


def test_cellkey_roundtrip_and_level_separation():
    rng = np.random.default_rng(0)
    lv = rng.integers(0, 4, 5000)
    ix = rng.integers(-3000, 3000, 5000)
    iy = rng.integers(-3000, 3000, 5000)
    k = pack_cell_key(lv, ix, iy)
    l2, x2, y2 = unpack_cell_key(k)
    assert np.array_equal(lv, l2) and np.array_equal(ix, x2) and np.array_equal(iy, y2)
    # same integer indices at different levels never collide
    same = pack_cell_key(np.arange(4), np.full(4, 17), np.full(4, -5))
    assert np.unique(same).size == 4


def test_no_key_collisions_in_projection(schedule):
    rng = np.random.default_rng(3)
    x = rng.uniform(-100, 100, 60000)
    y = rng.uniform(-100, 100, 60000)
    p = schedule.project(x, y)
    tuples = set(zip(p.level.tolist(), p.ix.tolist(), p.iy.tolist()))
    assert len(tuples) == np.unique(p.key).size


def test_every_point_inside_its_cell(schedule):
    rng = np.random.default_rng(4)
    x = rng.uniform(-100, 100, 50000)
    y = rng.uniform(-100, 100, 50000)
    p = schedule.project(x, y)
    xa, ya = x[p.accepted], y[p.accepted]
    x0, y0, x1, y1 = schedule.cell_bounds(p.level, p.ix, p.iy)
    eps = 1e-9
    assert np.all((x0 <= xa + eps) & (xa < x1 + eps) & (y0 <= ya + eps) & (ya < y1 + eps))


def test_partition_has_no_overlaps_or_gaps(schedule):
    """Allocated cells never overlap: each alignment tile holds exactly one level,
    and a dense allocation tiles each tile exactly (subdiv^2 cells of one size)."""
    rng = np.random.default_rng(5)
    n = 200000
    r = np.sqrt(rng.uniform(0, 100 ** 2, n))
    a = rng.uniform(0, 2 * np.pi, n)
    cells = aggregate_cells(schedule, r * np.cos(a), r * np.sin(a), np.zeros(n), np.zeros(n, np.int64))
    chk = check_partition(schedule, cells)
    assert chk["ok"] and chk["mixed_level_tiles"] == 0 and chk["misassigned_cells"] == 0
    # area covered by the cells of each tile equals the tile area when every child is occupied
    for b in schedule.bands:
        assert b.subdiv * b.cell_size_m == pytest.approx(schedule.tile_m)


def test_dense_counts_match_disk_area(schedule, uniform):
    dense_u = sum(uniform.dense_cell_counts())
    assert dense_u == pytest.approx(np.pi * 100 ** 2 / 0.05 ** 2, rel=1e-3)
    per_level = schedule.dense_cell_counts()
    # 60-100 m annulus at 0.5 m cells
    assert per_level[3] == pytest.approx(np.pi * (100 ** 2 - 60 ** 2) / 0.25, rel=0.01)


def test_custom_non_power_of_two_schedule():
    s = ResolutionSchedule([15, 40, 80], [0.04, 0.12, 0.3], 80)  # LCM 0.6 m tile, GCD 2 cm quantum
    assert s.tile_m == pytest.approx(0.6)
    assert s.quantum_m == pytest.approx(0.02)
    rng = np.random.default_rng(6)
    x = rng.uniform(-80, 80, 40000)
    y = rng.uniform(-80, 80, 40000)
    cells = aggregate_cells(s, x, y, np.zeros_like(x), np.zeros(x.size, np.int64))
    assert check_partition(s, cells)["ok"]
