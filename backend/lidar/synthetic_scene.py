"""Deterministic synthetic LiDAR scenes, produced by ray-casting a spinning multi-beam sensor.

The simulated sensor behaves like a 360 degree mechanical LiDAR: ``beams``
elevation channels (denser near the horizon) x azimuth steps. Every ray is
intersected with a terrain height field and with box / cylinder primitives,
so the output has the characteristics that motivate adaptive mapping:
concentric ground rings, occlusion shadows, and point density that falls off
with range.

Determinism: the static layout depends only on ``seed``; per-frame noise uses
``default_rng([seed, frame_index])``; actor motion is a closed-form function of
sensor time ``t = frame_index / 10 Hz``. The same (config, frame_index) always
produces the identical point cloud.

Ground is modelled as a base plane (optionally sloped along x) intersected
exactly, after which the hit is lifted to the terrain height field at that
x-y position (curbs, potholes, ditches, hills). Terrain self-occlusion is not
simulated; object occlusion is exact.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from backend.core.config import SceneConfig, SemanticClass as C
from backend.lidar.frame import PointFrame

SCAN_PERIOD_S = 0.1  # 10 Hz spin
MAX_SENSOR_RANGE = 120.0

# Reflectivity (calibrated intensity) of materials
I_ASPHALT, I_MARKING, I_CONCRETE, I_GRASS, I_DIRT = 0.14, 0.9, 0.42, 0.62, 0.24


# --------------------------------------------------------------------------- primitives
@dataclass
class Box:
    cx: float
    cy: float
    z0: float
    length: float
    width: float
    height: float
    yaw: float
    label: int
    inst: int
    intensity: float


@dataclass
class Cyl:
    cx: float
    cy: float
    z0: float
    radius: float
    height: float
    label: int
    inst: int
    intensity: float


@dataclass
class Actor:
    """A moving (or deliberately stationary) scene object with closed-form motion."""

    inst: int
    kind: str                 # car | truck | bus | moto | pedestrian
    label: int
    mode: str                 # lane_x | lane_y | cross_y | static | track
    p: dict = field(default_factory=dict)
    intensity: float = 0.4

    def state(self, t: float) -> tuple[float, float, float, float, float]:
        """(x, y, yaw, vx, vy) at time t."""
        p = self.p
        if self.mode == "lane_x":
            lo, hi = p["xmin"], p["xmax"]
            span = hi - lo
            x = lo + ((p["x0"] - lo + p["v"] * t) % span)
            return x, p["y"], 0.0 if p["v"] >= 0 else math.pi, p["v"], 0.0
        if self.mode == "lane_y":
            lo, hi = p["ymin"], p["ymax"]
            span = hi - lo
            y = lo + ((p["y0"] - lo + p["v"] * t) % span)
            return p["x"], y, math.copysign(math.pi / 2, p["v"]), 0.0, p["v"]
        if self.mode == "cross_y":  # walk back and forth between y0 and y1 (triangle wave)
            y0, y1, v = p["y0"], p["y1"], p["v"]
            L = abs(y1 - y0)
            s = (p.get("phase", 0.0) * 2 * L + v * t) % (2 * L)
            forward = s < L
            y = y0 + math.copysign(s if forward else 2 * L - s, y1 - y0)
            vy = math.copysign(v, y1 - y0) * (1 if forward else -1)
            return p["x"], y, math.copysign(math.pi / 2, vy), 0.0, vy
        if self.mode == "track":  # follow a curve y = f(x) with speed v along x
            lo, hi = p["xmin"], p["xmax"]
            x = lo + ((p["x0"] - lo + p["v"] * t) % (hi - lo))
            f: Callable[[float], float] = p["curve"]
            y = f(x)
            dy = (f(x + 0.5) - f(x - 0.5))
            yaw = math.atan2(dy, 1.0)
            return x, y, yaw, p["v"] * math.cos(yaw), p["v"] * math.sin(yaw)
        return p["x"], p["y"], p.get("yaw", 0.0), 0.0, 0.0


def actor_shapes(a: Actor, x: float, y: float, yaw: float, zg: float) -> list[Box | Cyl]:
    c, s = math.cos(yaw), math.sin(yaw)
    lab, inst, it = a.label, a.inst, a.intensity
    if a.kind == "pedestrian":
        return [Cyl(x, y, zg, 0.22, 1.72, lab, inst, it)]
    if a.kind == "moto":
        return [Box(x, y, zg + 0.3, 2.0, 0.75, 1.1, yaw, lab, inst, it)]
    if a.kind == "bus":
        return [Box(x, y, zg + 0.35, 11.5, 2.55, 2.9, yaw, lab, inst, it)]
    if a.kind == "truck":
        L = a.p.get("length", 8.0)
        cab_off = L / 2 - 1.1
        cargo_off = -1.2
        return [
            Box(x + c * cab_off, y + s * cab_off, zg + 0.5, 2.2, 2.45, 2.5, yaw, lab, inst, it),
            Box(x + c * cargo_off, y + s * cargo_off, zg + 0.9, L - 2.4, 2.5, 2.8, yaw, lab, inst, it),
        ]
    # passenger car: body + cabin
    L, W = a.p.get("length", 4.5), a.p.get("width", 1.85)
    off = -0.06 * L
    return [
        Box(x, y, zg + 0.3, L, W, 0.75, yaw, lab, inst, it),
        Box(x + c * off, y + s * off, zg + 1.05, 0.55 * L, 0.92 * W, 0.45, yaw, lab, inst, it),
    ]


# --------------------------------------------------------------------------- ground models
class RoadGround:
    """Straight road along x with lanes, markings, curbs, sidewalks, grass, optional cross street."""

    def __init__(self, yr: float, yl: float, dashed: list[float], solid: list[float], sidewalk_w: float = 4.0,
                 curb: float = 0.15, cross: tuple[float, float] | None = None,
                 crosswalk: tuple[float, float] | None = None, potholes: list[tuple] | None = None,
                 slope_x: float = 0.0, shoulder_is_gravel: bool = False):
        self.yr, self.yl = yr, yl
        self.dashed, self.solid = dashed, solid
        self.sw, self.curb = sidewalk_w, curb
        self.cross, self.crosswalk = cross, crosswalk
        self.potholes = potholes or []
        self.slope_x = slope_x
        self.gravel = shoulder_is_gravel

    def base_slope(self) -> float:
        return self.slope_x

    def eval(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        n = x.shape[0]
        z = self.slope_x * x
        label = np.full(n, int(C.DRIVABLE_TERRAIN), dtype=np.int8)
        inten = np.full(n, I_ASPHALT, dtype=np.float32)
        sigma = np.full(n, 0.004, dtype=np.float32)

        on_main = (y >= self.yr) & (y <= self.yl)
        dist = np.maximum(np.maximum(self.yr - y, y - self.yl), 0.0)
        on_road = on_main.copy()
        if self.cross:
            x0, x1 = self.cross
            on_cross = (x >= x0) & (x <= x1)
            on_road |= on_cross
            dist = np.minimum(dist, np.maximum(np.maximum(x0 - x, x - x1), 0.0))
        side = ~on_road & (dist <= self.sw)
        grass = ~on_road & (dist > self.sw)

        # lane markings (retro-reflective paint) only on the main carriageway outside the junction
        mark = np.zeros(n, dtype=bool)
        main_only = on_main if not self.cross else on_main & ~((x >= self.cross[0]) & (x <= self.cross[1]))
        for yl in self.dashed:
            mark |= main_only & (np.abs(y - yl) < 0.075) & (np.mod(x, 9.0) < 3.0)
        for yl in self.solid:
            mark |= main_only & (np.abs(y - yl) < 0.075)
        if self.crosswalk:
            cx0, cx1 = self.crosswalk
            mark |= on_main & (x >= cx0) & (x <= cx1) & (np.mod(y, 1.2) < 0.6)
            mark |= on_main & (np.abs(x - (cx0 - 1.5)) < 0.2) & (y < (self.yr + self.yl) / 2)  # stop line
        inten[mark] = I_MARKING

        z = np.where(side, z + self.curb, z)
        label[side] = C.NON_DRIVABLE_TERRAIN
        inten[side] = 0.45 if self.gravel else I_CONCRETE
        sigma[side] = 0.012 if self.gravel else 0.004

        und = 0.08 * np.sin(x / 7.0) * np.cos(y / 5.0)
        z = np.where(grass, z + self.curb + 0.05 + und, z)
        label[grass] = C.NON_DRIVABLE_TERRAIN
        inten[grass] = I_GRASS
        sigma[grass] = 0.04

        for (px, py, pr, pd) in self.potholes:
            d2 = ((x - px) ** 2 + (y - py) ** 2) / (pr * pr)
            inside = on_road & (d2 < 1.0)
            z = np.where(inside, z - pd * (1.0 - d2), z)
            label[inside] = C.NON_DRIVABLE_TERRAIN

        z = z + rng.normal(0.0, 1.0, n) * sigma
        inten = np.clip(inten + rng.normal(0.0, 0.035, n), 0.0, 1.0)
        return z, label, inten

    def height(self, x: float, y: float) -> float:
        z, _, _ = self.eval(np.array([x]), np.array([y]), np.random.default_rng(0))
        return float(z[0])


class TerrainGround:
    """Off-road: rolling hills, a smooth dirt track, a ditch and a steep embankment."""

    def __init__(self, amp: float = 1.0):
        self.amp = amp
        self._h0 = 0.0
        self._h0 = float(self._hills(np.array([0.0]), np.array([0.0]))[0])

    @staticmethod
    def track_y(x):
        return 6.0 * np.sin(np.asarray(x) / 35.0)

    def _hills(self, x, y):
        a = self.amp
        return (a * 0.9 * np.sin(x / 19.0 + 0.3) * np.cos(y / 23.0)
                + a * 0.55 * np.sin((x - 2.0 * y) / 13.0)) - self._h0

    def base_slope(self) -> float:
        return 0.0

    def eval(self, x, y, rng):
        n = x.shape[0]
        z = self._hills(x, y)
        dy = y - self.track_y(x)
        label = np.full(n, int(C.NON_DRIVABLE_TERRAIN), dtype=np.int8)
        inten = np.full(n, I_GRASS, dtype=np.float32)
        sigma = np.full(n, 0.05, dtype=np.float32)

        track = np.abs(dy) < 2.6
        label[track] = C.DRIVABLE_TERRAIN
        inten[track] = I_DIRT
        sigma[track] = 0.008

        ditch = (dy < -2.6) & (dy > -4.4)
        depth = 0.45 * np.sin(np.clip((-dy - 2.6) / 1.8, 0, 1) * math.pi)
        z = np.where(ditch, z - depth, z)
        sigma[ditch] = 0.02
        inten[ditch] = 0.3

        emb = y > 24.0
        z = np.where(emb, z + np.minimum(0.55 * (y - 24.0), 9.0), z)

        z = z + rng.normal(0.0, 1.0, n) * sigma
        inten = np.clip(inten + rng.normal(0.0, 0.035, n), 0.0, 1.0)
        return z, label, inten

    def height(self, x: float, y: float) -> float:
        z, _, _ = self.eval(np.array([x]), np.array([y]), np.random.default_rng(0))
        return float(z[0])


# --------------------------------------------------------------------------- scene layouts
@dataclass
class SceneLayout:
    ground: RoadGround | TerrainGround
    static: list[Box | Cyl]
    actors: list[Actor]
    static_meta: list[dict]


class _Ids:
    def __init__(self):
        self.n = 0

    def __call__(self) -> int:
        self.n += 1
        return self.n


def _pole(ids, meta, x, y, zg, h=7.0, r=0.12, arm_dir=0.0, arm_len=0.0):
    i = ids()
    shapes: list[Box | Cyl] = [Cyl(x, y, zg, r, h, C.STATIC_OBSTACLE, i, 0.5)]
    if arm_len > 0:  # overhanging lamp arm
        ay = y + arm_dir * arm_len / 2
        shapes.append(Box(x, ay, zg + h - 0.35, 0.18, arm_len, 0.18, 0.0, C.STATIC_OBSTACLE, i, 0.5))
    meta.append({"id": i, "kind": "pole", "class": int(C.STATIC_OBSTACLE), "x": x, "y": y})
    return shapes


def _tree(ids, meta, x, y, zg, canopy=3.2, trunk_h=2.7):
    i = ids()
    meta.append({"id": i, "kind": "tree", "class": int(C.STATIC_OBSTACLE), "x": x, "y": y})
    return [
        Cyl(x, y, zg, 0.18, trunk_h, C.STATIC_OBSTACLE, i, 0.35),
        Box(x, y, zg + trunk_h - 0.2, canopy, canopy, 2.4, 0.4, C.STATIC_OBSTACLE, i, 0.28),
    ]


def _buildings(ids, meta, rng, x0, x1, y_near, side, zg_fn):
    """Row of building facades between x0 and x1, starting at y_near and extending away from the road."""
    shapes: list[Box | Cyl] = []
    x = x0
    while x < x1 - 4:
        L = min(float(rng.uniform(12, 28)), x1 - x)
        D = float(rng.uniform(8, 14))
        H = float(rng.uniform(6, 18))
        cy = y_near + side * D / 2
        i = ids()
        shapes.append(Box(x + L / 2, cy, zg_fn(x + L / 2, y_near), L, D, H, 0.0, C.STATIC_OBSTACLE, i, 0.35))
        meta.append({"id": i, "kind": "building", "class": int(C.STATIC_OBSTACLE), "x": x + L / 2, "y": cy})
        x += L + float(rng.uniform(3, 6))
    return shapes


def _car(ids, kind="car", mode="static", **p) -> Actor:
    label = C.PEDESTRIAN if kind == "pedestrian" else C.VEHICLE
    return Actor(ids(), kind, int(label), mode, p, intensity=0.3 if kind == "pedestrian" else 0.45)


def build_layout(preset: str, seed: int) -> SceneLayout:
    rng = np.random.default_rng(seed)
    ids = _Ids()
    meta: list[dict] = []
    static: list[Box | Cyl] = []
    actors: list[Actor] = []

    if preset == "terrain":
        g = TerrainGround()
        zg = g.height
        for (x, y) in [(14, -9), (27, 12), (-18, -12), (40, -15), (55, 10), (-35, 9), (70, -8), (-60, -16)]:
            static += _tree(ids, meta, x, y, zg(x, y), canopy=float(rng.uniform(3, 5)))
        for _ in range(16):
            x, y = float(rng.uniform(-90, 90)), float(rng.uniform(-40, 22))
            if abs(y - float(g.track_y(x))) < 4.0 or math.hypot(x, y) < 5:
                continue
            s = float(rng.uniform(0.5, 1.4))
            i = ids()
            static.append(Box(x, y, zg(x, y) - 0.1, s * 1.3, s, s * 0.9, float(rng.uniform(0, 3)),
                              C.STATIC_OBSTACLE, i, 0.4))
            meta.append({"id": i, "kind": "rock", "class": int(C.STATIC_OBSTACLE), "x": x, "y": y})
        for (x0, x1, y) in [(-70, -40, -22), (30, 62, -26)]:  # dry-stone walls
            i = ids()
            static.append(Box((x0 + x1) / 2, y, zg((x0 + x1) / 2, y) - 0.2, x1 - x0, 0.6, 1.4, 0.05,
                              C.STATIC_OBSTACLE, i, 0.38))
            meta.append({"id": i, "kind": "wall", "class": int(C.STATIC_OBSTACLE), "x": (x0 + x1) / 2, "y": y})
        curve = lambda x: float(TerrainGround.track_y(x))  # noqa: E731
        actors.append(_car(ids, "car", "track", x0=16.0, v=4.5, xmin=12.0, xmax=98.0, curve=curve, length=4.2))
        actors.append(_car(ids, "car", "track", x0=-70.0, v=6.0, xmin=-98.0, xmax=-12.0, curve=curve, length=4.6))
        actors.append(_car(ids, "car", "static", x=-22.0, y=-7.5, yaw=0.3))
        actors.append(_car(ids, "pedestrian", "cross_y", x=24.0, y0=-6.0, y1=9.0, v=1.2))
        actors.append(_car(ids, "pedestrian", "static", x=-11.0, y=4.5))
        return SceneLayout(g, static, actors, meta)

    if preset == "road":
        g = RoadGround(yr=-1.75, yl=12.25, dashed=[1.75, 8.75], solid=[5.15, 5.35], sidewalk_w=3.0,
                       potholes=[(12.0, 0.9, 0.55, 0.16), (24.0, 3.6, 0.8, 0.2), (-15.0, 9.5, 0.6, 0.15),
                                 (48.0, -0.3, 0.7, 0.18)],
                       slope_x=0.015, shoulder_is_gravel=True)
        zg = g.height
        for x0 in range(-90, 91, 30):
            i = ids()
            static.append(Box(x0 + 12, -5.2, zg(x0 + 12, -5.2), 24.0, 0.25, 0.8, 0.0, C.STATIC_OBSTACLE, i, 0.5))
            meta.append({"id": i, "kind": "guardrail", "class": int(C.STATIC_OBSTACLE), "x": x0 + 12, "y": -5.2})
        for x in range(-80, 81, 32):
            static += _pole(ids, meta, float(x), -3.6, zg(x, -3.6), h=8.0, arm_dir=1.0, arm_len=2.2)
            static += _pole(ids, meta, float(x) + 16, 15.8, zg(x + 16, 15.8), h=8.0, arm_dir=-1.0, arm_len=2.2)
        # overhead sign gantry spanning the carriageway (overhanging obstacle)
        i = ids()
        static += [Cyl(45.0, -4.2, zg(45, -4.2), 0.25, 6.3, C.STATIC_OBSTACLE, i, 0.5),
                   Cyl(45.0, 16.6, zg(45, 16.6), 0.25, 6.3, C.STATIC_OBSTACLE, i, 0.5),
                   Box(45.0, 6.2, zg(45, 6) + 5.3, 1.0, 21.0, 1.1, 0.0, C.STATIC_OBSTACLE, i, 0.6)]
        meta.append({"id": i, "kind": "gantry", "class": int(C.STATIC_OBSTACLE), "x": 45.0, "y": 6.2})
        for _ in range(14):
            x, y = float(rng.uniform(-90, 90)), float(rng.choice([rng.uniform(-30, -9), rng.uniform(20, 40)]))
            static += _tree(ids, meta, x, y, zg(x, y), canopy=float(rng.uniform(3, 5)))
        actors += [
            _car(ids, "car", "lane_x", x0=-30.0, y=3.5, v=16.0, xmin=-100.0, xmax=100.0),
            _car(ids, "truck", "lane_x", x0=35.0, y=3.5, v=13.0, xmin=-100.0, xmax=100.0, length=9.0),
            _car(ids, "car", "lane_x", x0=20.0, y=7.0, v=-18.0, xmin=-100.0, xmax=100.0),
            _car(ids, "car", "lane_x", x0=-45.0, y=10.5, v=-14.0, xmin=-100.0, xmax=100.0, length=4.8),
            _car(ids, "bus", "lane_x", x0=70.0, y=10.5, v=-11.0, xmin=-100.0, xmax=100.0),
            _car(ids, "car", "static", x=-11.0, y=0.0),
            _car(ids, "pedestrian", "lane_x", x0=-20.0, y=-2.6, v=1.4, xmin=-40.0, xmax=40.0),
        ]
        return SceneLayout(g, static, actors, meta)

    if preset == "mixed_traffic":
        g = RoadGround(yr=-1.75, yl=20.25, dashed=[1.75, 5.25, 13.25, 16.75], solid=[9.0, 9.75],
                       sidewalk_w=4.0, crosswalk=(18.0, 22.0),
                       potholes=[(-12.0, 2.5, 0.6, 0.15)])
        zg = g.height
        for x0 in (-80.0, -40.0, 30.0, 60.0):  # median jersey barrier with gaps
            i = ids()
            static.append(Box(x0 + 12, 9.375, zg(x0 + 12, 9.4), 24.0, 0.5, 0.85, 0.0, C.STATIC_OBSTACLE, i, 0.55))
            meta.append({"id": i, "kind": "barrier", "class": int(C.STATIC_OBSTACLE), "x": x0 + 12, "y": 9.375})
        for x in range(-84, 85, 24):
            static += _pole(ids, meta, float(x), -3.3, zg(x, -3.3), h=7.5, arm_dir=1.0, arm_len=1.8)
            static += _pole(ids, meta, float(x) + 12, 22.0, zg(x + 12, 22.0), h=7.5, arm_dir=-1.0, arm_len=1.8)
        static += _buildings(ids, meta, rng, -95, 95, -7.0, -1, zg)
        static += _buildings(ids, meta, rng, -95, 95, 26.0, 1, zg)
        lanes = [(3.5, 1), (7.0, 1), (11.5, -1), (15.0, -1), (18.5, -1)]
        kinds = ["car", "car", "truck", "car", "moto", "bus", "car", "car", "car", "car", "moto", "car"]
        k = 0
        for (ly, d) in lanes:
            for j in range(2 if ly != 7.0 else 3):
                kind = kinds[k % len(kinds)]
                k += 1
                speed = d * float(rng.uniform(7, 15))
                actors.append(_car(ids, kind, "lane_x", x0=float(-80 + 70 * j + rng.uniform(0, 30)), y=ly,
                                   v=speed, xmin=-100.0, xmax=100.0, length=float(rng.uniform(4.2, 4.9))))
        actors.append(_car(ids, "car", "static", x=10.0, y=0.0))    # queued ahead of ego
        actors.append(_car(ids, "car", "static", x=-8.5, y=0.0))    # queued behind ego
        for j, ph in enumerate([0.0, 0.2, 0.45, 0.7, 0.9]):
            actors.append(_car(ids, "pedestrian", "cross_y", x=19.0 + 0.8 * j, y0=-4.5, y1=23.5,
                               v=float(rng.uniform(1.0, 1.6)), phase=ph))
        actors.append(_car(ids, "pedestrian", "lane_x", x0=-30.0, y=-4.2, v=1.3, xmin=-60.0, xmax=60.0))
        return SceneLayout(g, static, actors, meta)

    # ---------------------------------------------------------------- urban (default)
    g = RoadGround(yr=-4.25, yl=12.25, dashed=[1.75, 8.75], solid=[5.15, 5.35], sidewalk_w=4.0,
                   cross=(34.0, 46.0), crosswalk=(28.0, 32.0), potholes=[(-6.0, 3.2, 0.5, 0.15), (18.0, 9.0, 0.7, 0.18)])
    zg = g.height
    static += _buildings(ids, meta, rng, -95, 29.5, -8.75, -1, zg)
    static += _buildings(ids, meta, rng, 50.5, 95, -8.75, -1, zg)
    static += _buildings(ids, meta, rng, -95, 29.5, 16.75, 1, zg)
    static += _buildings(ids, meta, rng, 50.5, 95, 16.75, 1, zg)
    for x in range(-84, 85, 20):
        if 26 <= x <= 50:
            continue
        static += _pole(ids, meta, float(x), -5.0, zg(x, -5.0), h=7.0, arm_dir=1.0, arm_len=2.0)
        static += _pole(ids, meta, float(x) + 10, 14.0, zg(x + 10, 14.0), h=7.0, arm_dir=-1.0, arm_len=2.0)
    for x in (-50.0, -30.0, 3.0, 15.0, 62.0, 78.0):
        static += _tree(ids, meta, x, -7.2, zg(x, -7.2), canopy=3.4)
    # bus shelter: roof (overhang) + two posts
    i = ids()
    static += [Box(-18.0, -7.0, zg(-18, -7) + 2.4, 4.0, 1.8, 0.15, 0.0, C.STATIC_OBSTACLE, i, 0.45),
               Cyl(-19.8, -7.6, zg(-19.8, -7.6), 0.06, 2.4, C.STATIC_OBSTACLE, i, 0.5),
               Cyl(-16.2, -7.6, zg(-16.2, -7.6), 0.06, 2.4, C.STATIC_OBSTACLE, i, 0.5)]
    meta.append({"id": i, "kind": "bus_shelter", "class": int(C.STATIC_OBSTACLE), "x": -18.0, "y": -7.0})
    for x in (26.0, 26.9, 27.8):  # bollards
        i = ids()
        static.append(Cyl(x, -5.0, zg(x, -5.0), 0.1, 0.9, C.STATIC_OBSTACLE, i, 0.55))
        meta.append({"id": i, "kind": "bollard", "class": int(C.STATIC_OBSTACLE), "x": x, "y": -5.0})
    for x in (-40.0, -33.0, -26.0, 7.0, 14.5, 21.0, 56.0, 63.0, 71.0):  # parked cars
        actors.append(_car(ids, "car", "static", x=x, y=-3.0, length=float(rng.uniform(4.2, 4.8))))
    actors.append(_car(ids, "car", "static", x=9.5, y=0.0))               # stopped ahead of ego
    actors.append(_car(ids, "car", "static", x=-8.5, y=0.0))              # stopped behind ego
    actors.append(_car(ids, "car", "static", x=40.0, y=-14.0, yaw=math.pi / 2))  # cross street, waiting
    actors.append(_car(ids, "car", "static", x=37.0, y=24.0, yaw=-math.pi / 2))
    actors += [
        _car(ids, "car", "lane_x", x0=-60.0, y=3.5, v=10.0, xmin=-100.0, xmax=100.0),
        _car(ids, "truck", "lane_x", x0=15.0, y=3.5, v=8.0, xmin=-100.0, xmax=100.0, length=7.5),
        _car(ids, "car", "lane_x", x0=40.0, y=7.0, v=-11.0, xmin=-100.0, xmax=100.0),
        _car(ids, "bus", "lane_x", x0=-20.0, y=10.5, v=-7.5, xmin=-100.0, xmax=100.0),
        _car(ids, "car", "lane_x", x0=80.0, y=10.5, v=-9.0, xmin=-100.0, xmax=100.0),
        _car(ids, "pedestrian", "cross_y", x=29.5, y0=-5.5, y1=13.5, v=1.3, phase=0.0),
        _car(ids, "pedestrian", "cross_y", x=31.0, y0=13.5, y1=-5.5, v=1.1, phase=0.3),
        _car(ids, "pedestrian", "lane_x", x0=-30.0, y=-6.0, v=1.4, xmin=-60.0, xmax=26.0),
        _car(ids, "pedestrian", "lane_x", x0=10.0, y=14.8, v=-1.2, xmin=-60.0, xmax=28.0),
        _car(ids, "pedestrian", "static", x=-14.4, y=-5.6),                # waiting at bus stop
        _car(ids, "pedestrian", "lane_x", x0=60.0, y=-6.2, v=1.0, xmin=52.0, xmax=90.0),
    ]
    return SceneLayout(g, static, actors, meta)


# --------------------------------------------------------------------------- ray casting
def beam_elevations_deg(n: int) -> np.ndarray:
    """Beam elevations: a few upward channels, the rest concentrated near the horizon (HDL-64 like)."""
    n_up = max(2, int(round(n * 0.09)))
    up = np.linspace(3.0, 0.2, n_up)
    v = np.linspace(0.0, 1.0, n - n_up)
    down = -0.5 - 24.5 * v ** 1.7
    return np.concatenate([up, down])


@lru_cache(maxsize=8)
def ray_directions(beams: int, az_res_deg: float):
    elev = np.deg2rad(beam_elevations_deg(beams))
    n_az = int(round(360.0 / az_res_deg))
    az = -math.pi + (np.arange(n_az) + 0.5) * (2 * math.pi / n_az)
    ce, se = np.cos(elev)[:, None], np.sin(elev)[:, None]
    dx = ce * np.cos(az)[None, :]
    dy = ce * np.sin(az)[None, :]
    dz = np.broadcast_to(se, dx.shape).copy()
    return elev, az, dx, dy, dz


def _az_columns(n_az: int, az_center: float, half_span: float) -> np.ndarray:
    step = 2 * math.pi / n_az
    j0 = int(math.floor((az_center - half_span + math.pi) / step)) - 1
    j1 = int(math.ceil((az_center + half_span + math.pi) / step)) + 1
    if j1 - j0 >= n_az:
        return np.arange(n_az)
    return np.arange(j0, j1 + 1) % n_az


def _beam_rows(elev: np.ndarray, sz: float, z0: float, z1: float, dmin: float, dmax: float) -> np.ndarray:
    dmin = max(dmin, 0.2)
    lo = min(math.atan2(z0 - sz, dmin), math.atan2(z0 - sz, dmax))
    hi = max(math.atan2(z1 - sz, dmin), math.atan2(z1 - sz, dmax))
    m = math.radians(0.3)
    return np.flatnonzero((elev >= lo - m) & (elev <= hi + m))


def _intersect_box(b: Box, dx, dy, dz, sz: float):
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    ox, oy, oz = -b.cx, -b.cy, sz - (b.z0 + b.height / 2)
    lx, ly = c * ox + s * oy, -s * ox + c * oy
    ddx, ddy = c * dx + s * dy, -s * dx + c * dy
    tmin = np.full(dx.shape, -np.inf)
    tmax = np.full(dx.shape, np.inf)
    for o, d, h in ((lx, ddx, b.length / 2), (ly, ddy, b.width / 2), (oz, dz, b.height / 2)):
        d = np.where(np.abs(d) < 1e-12, 1e-12, d)
        t1 = (-h - o) / d
        t2 = (h - o) / d
        tmin = np.maximum(tmin, np.minimum(t1, t2))
        tmax = np.minimum(tmax, np.maximum(t1, t2))
    hit = (tmax >= tmin) & (tmin > 0.05)
    return np.where(hit, tmin, np.inf)


def _intersect_cyl(cy: Cyl, dx, dy, dz, sz: float):
    ox, oy = -cy.cx, -cy.cy
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    cc = ox * ox + oy * oy - cy.radius ** 2
    disc = b * b - 4 * a * cc
    ok = (disc >= 0) & (a > 1e-12)
    t = np.where(ok, (-b - np.sqrt(np.where(ok, disc, 0))) / (2 * np.where(a > 1e-12, a, 1)), np.inf)
    z = sz + t * dz
    side = ok & (t > 0.05) & (z >= cy.z0) & (z <= cy.z0 + cy.height)
    t_side = np.where(side, t, np.inf)
    # top cap, seen from above
    z1 = cy.z0 + cy.height
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = (z1 - sz) / dz
    cap = (dz < 0) & (tc > 0.05)
    hx, hy = tc * dx - cy.cx, tc * dy - cy.cy
    cap &= hx * hx + hy * hy <= cy.radius ** 2
    return np.minimum(t_side, np.where(cap, tc, np.inf))


def _shape_extent(sh: Box | Cyl):
    if isinstance(sh, Cyl):
        d = math.hypot(sh.cx, sh.cy)
        return d, sh.radius, sh.z0, sh.z0 + sh.height
    d = math.hypot(sh.cx, sh.cy)
    return d, 0.5 * math.hypot(sh.length, sh.width), sh.z0, sh.z0 + sh.height


class SyntheticLidar:
    """Frame source for synthetic scenes."""

    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg
        self.layout = build_layout(cfg.preset, cfg.seed)

    def actor_states(self, t: float) -> list[dict]:
        out = []
        for a in self.layout.actors:
            x, y, yaw, vx, vy = a.state(t)
            out.append({"id": a.inst, "kind": a.kind, "class": a.label, "x": x, "y": y, "yaw": yaw,
                        "vx": vx, "vy": vy, "moving": math.hypot(vx, vy) > 0.05})
        return out

    def frame(self, frame_index: int) -> PointFrame:
        t0 = time.perf_counter()
        cfg = self.cfg
        t = frame_index * SCAN_PERIOD_S
        rng = np.random.default_rng([cfg.seed, frame_index])
        sz = cfg.sensor_height_m
        g = self.layout.ground
        elev, az, dx, dy, dz = ray_directions(cfg.beams, round(cfg.azimuth_resolution_deg, 4))
        n_b, n_az = dx.shape

        # ---- ground: exact intersection with the (optionally sloped) base plane
        slope = g.base_slope()
        denom = slope * dx - dz
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hit = np.where(denom > 1e-6, sz / denom, np.inf)
        t_hit[t_hit > MAX_SENSOR_RANGE * 1.05] = np.inf
        label = np.full(dx.shape, -1, dtype=np.int16)   # -1 = ground (resolved later)
        inst = np.full(dx.shape, -1, dtype=np.int32)
        inten = np.zeros(dx.shape, dtype=np.float32)

        # ---- objects
        shapes: list[Box | Cyl] = list(self.layout.static)
        actor_states = self.actor_states(t)
        for a, st in zip(self.layout.actors, actor_states):
            shapes += actor_shapes(a, st["x"], st["y"], st["yaw"], g.height(st["x"], st["y"]))
        for sh in shapes:
            d, rad, z0, z1 = _shape_extent(sh)
            if d - rad > MAX_SENSOR_RANGE:
                continue
            if d <= rad + 0.3:
                cols = np.arange(n_az)
            else:
                cols = _az_columns(n_az, math.atan2(sh.cy, sh.cx), math.asin(min(1.0, rad / d)))
            rows = _beam_rows(elev, sz, z0, z1, d - rad, d + rad)
            if rows.size == 0:
                continue
            ix = np.ix_(rows, cols)
            sdx, sdy, sdz = dx[ix], dy[ix], dz[ix]
            tt = _intersect_box(sh, sdx, sdy, sdz, sz) if isinstance(sh, Box) else _intersect_cyl(sh, sdx, sdy, sdz, sz)
            cur = t_hit[ix]
            better = tt < cur
            if not better.any():
                continue
            t_hit[ix] = np.where(better, tt, cur)
            label[ix] = np.where(better, sh.label, label[ix])
            inst[ix] = np.where(better, sh.inst, inst[ix])
            inten[ix] = np.where(better, sh.intensity, inten[ix])

        # ---- collect returns
        valid = np.isfinite(t_hit)
        valid &= rng.random(dx.shape) >= cfg.dropout
        r_idx, c_idx = np.nonzero(valid)
        tr = t_hit[r_idx, c_idx] + rng.normal(0.0, cfg.range_noise_m, r_idx.shape[0])
        px = tr * dx[r_idx, c_idx]
        py = tr * dy[r_idx, c_idx]
        pz = sz + tr * dz[r_idx, c_idx]
        lab = label[r_idx, c_idx].astype(np.int8)
        ins = inst[r_idx, c_idx]
        it = inten[r_idx, c_idx] + rng.normal(0.0, 0.04, r_idx.shape[0]).astype(np.float32)

        gm = lab < 0
        if gm.any():
            gz, glab, gint = g.eval(px[gm], py[gm], rng)
            pz[gm] = gz
            lab[gm] = glab
            it[gm] = gint
        keep = np.hypot(px, py) <= MAX_SENSOR_RANGE
        px, py, pz, lab, ins, it = px[keep], py[keep], pz[keep], lab[keep], ins[keep], it[keep]
        ts = t + (az[c_idx[keep]] + math.pi) / (2 * math.pi) * SCAN_PERIOD_S

        # ---- sensor noise: isolated spurious returns (dust / multipath)
        n_out = int(cfg.outlier_ratio * px.shape[0])
        if n_out:
            rr = rng.uniform(3.0, 60.0, n_out)
            aa = rng.uniform(-math.pi, math.pi, n_out)
            px = np.r_[px, rr * np.cos(aa)]
            py = np.r_[py, rr * np.sin(aa)]
            pz = np.r_[pz, rng.uniform(0.6, 6.0, n_out)]
            lab = np.r_[lab, np.full(n_out, int(C.UNKNOWN), np.int8)]
            ins = np.r_[ins, np.full(n_out, -1, np.int32)]
            it = np.r_[it, rng.uniform(0.0, 0.2, n_out).astype(np.float32)]
            ts = np.r_[ts, np.full(n_out, t)]

        xyz = np.stack([px, py, pz], axis=1).astype(np.float32)
        gt_objects = [dict(o, moving=False, vx=0.0, vy=0.0) for o in self.layout.static_meta] + actor_states
        return PointFrame(
            xyz=xyz,
            intensity=np.clip(it, 0.0, 1.0).astype(np.float32),
            timestamp=ts.astype(np.float64),
            frame_index=frame_index,
            sensor_time=t,
            source=f"synthetic:{cfg.preset}",
            gt_label=lab,
            gt_instance=ins,
            gt_objects=gt_objects,
            sim_ms=(time.perf_counter() - t0) * 1e3,
        )


PRESETS = [
    {"id": "urban", "label": "Cluttered urban", "description": "Buildings, parked cars, junction, crosswalk, bus shelter, trees."},
    {"id": "road", "label": "Road / highway", "description": "Sloped carriageway, potholes, guardrails, overhead sign gantry."},
    {"id": "mixed_traffic", "label": "Mixed dynamic traffic", "description": "6-lane road, many moving vehicles, pedestrians crossing."},
    {"id": "terrain", "label": "Challenging terrain", "description": "Off-road hills, dirt track, ditch, steep embankment, rocks, trees."},
]
