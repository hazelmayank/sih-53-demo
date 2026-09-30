"""Prototype multi-object tracker.

Greedy nearest-neighbour association with distance gating on constant-velocity
predictions, persistent track IDs, smoothed velocity and a motion state.
This is a transparent prototype tracker, not a validated production MOT
(no Kalman covariance, no Hungarian assignment, no re-identification).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from backend.core.config import SemanticClass as C

CLASS_GATE_SCALE = {int(C.PEDESTRIAN): 0.6, int(C.VEHICLE): 1.2}
MIN_HITS_FOR_MOTION = 3
VELOCITY_ALPHA = 0.45
NET_WINDOW = 6  # frames used for the net-displacement motion test


@dataclass
class Track:
    id: int
    cls: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    misses: int = 0
    age: int = 1
    last_time: float = 0.0
    trail: deque = field(default_factory=lambda: deque(maxlen=25))

    times: deque = field(default_factory=lambda: deque(maxlen=25))

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    def net_speed(self) -> float:
        """Speed from net displacement over the last few frames. Centroid jitter from
        changing viewpoints / partial occlusion largely cancels out, real motion does not."""
        k = min(len(self.trail), NET_WINDOW)
        if k < 2:
            return 0.0
        (x0, y0), (x1, y1) = self.trail[-k], self.trail[-1]
        dt = self.times[-1] - self.times[-k]
        return float(np.hypot(x1 - x0, y1 - y0) / dt) if dt > 0 else 0.0


@dataclass
class Detection:
    x: float
    y: float
    cls: int
    index: int  # index into the frame's object list


class NearestNeighbourTracker:
    def __init__(self, gate_m: float = 2.5, motion_threshold: float = 0.7, max_misses: int = 3):
        self.gate_m = gate_m
        self.motion_threshold = motion_threshold
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_time: float | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self._last_time = None

    def motion_speed(self, t: Track) -> float:
        # Semantic prior: STATIC_OBSTACLE is static by definition (walls, poles, trees).
        if t.hits < MIN_HITS_FOR_MOTION or t.cls == C.STATIC_OBSTACLE:
            return 0.0
        return min(t.speed, t.net_speed())

    def dynamic_score(self, t: Track) -> float:
        v = self.motion_speed(t)
        return float(np.clip((v - 0.5 * self.motion_threshold) / self.motion_threshold, 0.0, 1.0))

    def is_dynamic(self, t: Track) -> bool:
        return self.motion_speed(t) > self.motion_threshold

    def match(self, dets: list[Detection]) -> dict[int, Track]:
        """Associate detections with current tracks without changing tracker state
        (used to re-render the same frame after a configuration change)."""
        out: dict[int, Track] = {}
        used: set[int] = set()
        for d in dets:
            best, best_d = None, None
            for t in self.tracks:
                if t.id in used or not (d.cls == t.cls or C.UNKNOWN in (d.cls, t.cls)):
                    continue
                dist = float(np.hypot(d.x - t.x, d.y - t.y))
                if dist <= self.gate_m * CLASS_GATE_SCALE.get(t.cls, 1.0) and (best_d is None or dist < best_d):
                    best, best_d = t, dist
            if best is not None:
                used.add(best.id)
                out[d.index] = best
        return out

    def update(self, dets: list[Detection], timestamp: float) -> dict[int, Track]:
        """Associate detections; returns {detection index -> track}."""
        dt = 0.1 if self._last_time is None else max(timestamp - self._last_time, 1e-3)
        if self._last_time is not None and timestamp < self._last_time:  # sequence restarted
            self.reset()
            dt = 0.1
        self._last_time = timestamp

        pairs: list[tuple[float, int, int]] = []
        if self.tracks and dets:
            # vectorised gating: predicted track positions vs detections
            tp = np.array([(t.x + t.vx * dt, t.y + t.vy * dt) for t in self.tracks])
            tc = np.array([t.cls for t in self.tracks])
            dp = np.array([(d.x, d.y) for d in dets])
            dc = np.array([d.cls for d in dets])
            dist = np.hypot(tp[:, None, 0] - dp[None, :, 0], tp[:, None, 1] - dp[None, :, 1])
            gate = self.gate_m * np.array([CLASS_GATE_SCALE.get(int(c), 1.0) for c in tc])
            compatible = (tc[:, None] == dc[None, :]) | (tc[:, None] == C.UNKNOWN) | (dc[None, :] == C.UNKNOWN)
            ti_idx, di_idx = np.nonzero(compatible & (dist <= gate[:, None]))
            order = np.argsort(dist[ti_idx, di_idx], kind="stable")
            pairs = [(float(dist[ti_idx[k], di_idx[k]]), int(ti_idx[k]), int(di_idx[k])) for k in order]
        used_t, used_d = set(), set()
        assigned: dict[int, Track] = {}
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr, d = self.tracks[ti], dets[di]
            vx, vy = (d.x - tr.x) / dt, (d.y - tr.y) / dt
            a = 1.0 if tr.hits == 1 else VELOCITY_ALPHA
            tr.vx = a * vx + (1 - a) * tr.vx
            tr.vy = a * vy + (1 - a) * tr.vy
            tr.x, tr.y = d.x, d.y
            if d.cls != C.UNKNOWN:
                tr.cls = d.cls
            tr.hits += 1
            tr.misses = 0
            tr.age += 1
            tr.last_time = timestamp
            tr.trail.append((round(d.x, 2), round(d.y, 2)))
            tr.times.append(timestamp)
            assigned[d.index] = tr

        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
                tr.age += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for di, d in enumerate(dets):
            if di in used_d:
                continue
            tr = Track(id=self._next_id, cls=d.cls, x=d.x, y=d.y, last_time=timestamp)
            tr.trail.append((round(d.x, 2), round(d.y, 2)))
            tr.times.append(timestamp)
            self._next_id += 1
            self.tracks.append(tr)
            assigned[d.index] = tr
        return assigned
