"""Band-boundary behaviour at exactly 10 m, 30 m, 60 m and 100 m.

Convention (documented in adaptive_grid.py): band lookup uses half-open
intervals [lower, upper); a point's level is the level of its 0.5 m alignment
tile, chosen by the range of the tile centre; acceptance uses the point range
with an inclusive outer bound (r <= max_range).
"""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "r, level",
    [(0.0, 0), (9.999, 0), (10.0, 1), (29.999, 1), (30.0, 2), (59.999, 2), (60.0, 3), (99.999, 3), (100.0, 3), (140.0, 3)],
)
def test_level_for_range_half_open(schedule, r, level):
    assert int(schedule.level_for_range(r)) == level


def test_cell_size_for_range(schedule):
    r = np.array([5.0, 9.999, 10.0, 29.999, 30.0, 59.999, 60.0, 100.0])
    assert np.allclose(schedule.cell_size_for_range(r), [0.05, 0.05, 0.10, 0.10, 0.25, 0.25, 0.50, 0.50])


@pytest.mark.parametrize(
    "r, level",
    [(9.999, 0), (10.0, 1), (29.999, 1), (30.0, 2), (59.999, 2), (60.0, 3), (100.0, 3)],
)
@pytest.mark.parametrize("axis", ["x", "y"])
def test_points_on_axes_map_to_expected_band(schedule, r, level, axis):
    x, y = (np.array([r]), np.array([0.0])) if axis == "x" else (np.array([0.0]), np.array([r]))
    p = schedule.project(x, y)
    assert p.accepted.all()
    assert int(p.level[0]) == level


def test_exact_boundary_is_deterministic(schedule):
    x = np.full(1000, 10.0)
    y = np.zeros(1000)
    p = schedule.project(x, y)
    assert np.unique(p.key).size == 1  # identical input -> identical CellKey every time
    assert int(p.level[0]) == 1 and int(p.ix[0]) == 100 and int(p.iy[0]) == 0  # 10.0 / 0.10 = 100


def test_point_at_100m_accepted_beyond_rejected(schedule):
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    x = 100.0 * np.cos(angles)
    y = 100.0 * np.sin(angles)
    # snap to exactly 100 m in float64
    r = np.hypot(x, y)
    x, y = x * (100.0 / r), y * (100.0 / r)
    inside = schedule.project(x[np.hypot(x, y) <= 100.0], y[np.hypot(x, y) <= 100.0])
    assert inside.accepted.all()
    assert (inside.level == 3).all()
    out = schedule.project(np.array([100.001, 0.0, -150.0]), np.array([0.0, 100.0001, 0.0]))
    assert not out.accepted.any()


def test_level_follows_tile_centre(schedule):
    """Every point takes the level of its alignment tile (staircase boundary, no mixing)."""
    rng = np.random.default_rng(1)
    r = rng.uniform(8.0, 12.0, 20000)
    a = rng.uniform(0, 2 * np.pi, r.size)
    x, y = r * np.cos(a), r * np.sin(a)
    p = schedule.project(x, y)
    t = schedule.tile_m
    tx, ty = np.floor(x / t), np.floor(y / t)
    assert np.array_equal(p.level, schedule.tile_level(tx, ty))


def test_float_representation_at_cell_edges(schedule):
    # values that are not exactly representable must still floor onto the documented cell
    xs = np.array([0.15, 0.3, 0.45, 9.95, 29.9, 59.75])
    p = schedule.project(xs, np.zeros_like(xs))
    size = schedule.sizes[p.level]
    assert np.all(p.ix * size <= xs + 1e-9)
    assert np.all(xs < (p.ix + 1) * size + 1e-9)
    assert list(p.ix[:3]) == [3, 6, 9]  # 0.15/0.05 = 3 exactly by convention
