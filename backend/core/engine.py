"""Stream engine: owns the frame source, the pipeline, playback state and frame history,
and fans processed frames out to WebSocket subscribers."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from typing import Any

from backend.core.benchmark import run_benchmark
from backend.core.config import AppConfig, deep_merge
from backend.core.metrics import FpsMeter, MetricsHistory, process_rss_mb
from backend.core.pipeline import PerceptionPipeline, ProcessedFrame
from backend.lidar.frame import PointFrame
from backend.lidar.io import FileSequence
from backend.lidar.synthetic_scene import SyntheticLidar


class FrameSource:
    def __init__(self, cfg: AppConfig):
        if cfg.scene.source == "file":
            if not cfg.scene.file_sequence:
                raise ValueError("scene.source=file requires scene.file_sequence")
            self.impl: Any = FileSequence(cfg.scene.file_sequence, loop=cfg.stream.loop_file_sequence)
            self.length: int | None = len(self.impl)
        else:
            self.impl = SyntheticLidar(cfg.scene)
            self.length = None
        self.z_offset = cfg.scene.import_z_offset_m if cfg.scene.source == "file" else 0.0

    def frame(self, i: int) -> PointFrame | None:
        f = self.impl.frame(i)
        if f is not None and self.z_offset:
            f.xyz[:, 2] += self.z_offset
        return f


class Subscriber:
    def __init__(self, options: dict | None = None):
        self.options = {"points": True, "uniform": False, "max_points": 60000, **(options or {})}
        self.event = asyncio.Event()
        self.pending_frame: int | None = None  # explicit frame request (scrubbing)


class StreamEngine:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig()
        self._lock = threading.RLock()
        self.pipeline = PerceptionPipeline(self.cfg)
        self.source = FrameSource(self.cfg)
        self.frame_index = 0
        self.running = False
        self.started_once = False
        self.history: OrderedDict[int, ProcessedFrame] = OrderedDict()
        self.latest: ProcessedFrame | None = None
        self.last_raw: PointFrame | None = None
        self.fps = FpsMeter()
        self.metrics = MetricsHistory()
        self.subscribers: set[Subscriber] = set()
        self.error: str | None = None
        self.config_version = 0
        self._task: asyncio.Task | None = None
        self.started_at = time.time()
        self.benchmark_state: dict = {"running": False, "progress": 0.0, "message": "", "result": None}

    # ------------------------------------------------------------------ state
    def status(self) -> dict:
        return {
            "running": self.running,
            "frame_index": self.frame_index,
            "latest_frame_id": self.latest.frame_id if self.latest else None,
            "history": list(self.history.keys()),
            "source": self.cfg.scene.source,
            "preset": self.cfg.scene.preset,
            "sequence_length": self.source.length,
            "segmenter": self.pipeline.segmenter.info(),
            "config_version": self.config_version,
            "error": self.error,
            "subscribers": len(self.subscribers),
            "uptime_s": time.time() - self.started_at,
            "rss_mb": process_rss_mb(),
        }

    # ------------------------------------------------------------------ processing
    def _produce(self, advance: bool = True) -> ProcessedFrame | None:
        with self._lock:
            raw = self.source.frame(self.frame_index)
            if raw is None:
                return None
            pf = self.pipeline.process(raw)
            self.last_raw = raw
            if advance:
                self.frame_index += 1
            return pf

    def _reprocess_current(self) -> ProcessedFrame | None:
        """Re-run the pipeline on the last raw frame (after a config change while paused)."""
        with self._lock:
            if self.last_raw is None:
                return None
            return self.pipeline.process(self.last_raw, update_tracker=False)

    def _publish(self, pf: ProcessedFrame, count_fps: bool = True) -> None:
        pf.metrics["fps"] = self.fps.tick() if count_fps else self.fps.value()
        pf.metrics["rss_mb"] = process_rss_mb()
        self.latest = pf
        self.history[pf.frame_id] = pf
        while len(self.history) > self.cfg.stream.history_frames:
            self.history.popitem(last=False)
        self.metrics.add(pf.frame_id, pf.metrics)
        for s in self.subscribers:
            s.event.set()

    async def _loop(self) -> None:
        try:
            while self.running:
                t0 = time.perf_counter()
                try:
                    pf = await asyncio.to_thread(self._produce)
                except Exception as exc:  # keep the server alive, surface the error in the UI
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.running = False
                    self._notify()
                    break
                if pf is None:  # end of a non-looping file sequence
                    self.running = False
                    self._notify()
                    break
                self.error = None
                self._publish(pf)
                period = 1.0 / (self.cfg.stream.target_hz * self.cfg.stream.playback_speed)
                await asyncio.sleep(max(0.0, period - (time.perf_counter() - t0)))
        finally:
            self._task = None

    def _notify(self) -> None:
        for s in self.subscribers:
            s.event.set()

    # ------------------------------------------------------------------ controls
    async def start(self) -> None:
        self.started_once = True
        if self.running:
            return
        self.running = True
        self.fps.reset()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self.running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:  # pragma: no cover
                pass
        self._notify()

    async def step(self) -> ProcessedFrame | None:
        if self.running:
            return self.latest
        pf = await asyncio.to_thread(self._produce)
        if pf:
            self._publish(pf, count_fps=False)
        return pf

    async def reset(self) -> None:
        was_running = self.running
        await self.stop()
        with self._lock:
            self.frame_index = 0
            self.pipeline.reset()
            self.history.clear()
            self.metrics.clear()
            self.fps.reset()
        await self.step()
        if was_running:
            await self.start()

    async def seek(self, frame_index: int) -> ProcessedFrame | None:
        """Jump to an arbitrary frame (tracker restarts there)."""
        was_running = self.running
        await self.stop()
        with self._lock:
            self.frame_index = max(0, frame_index)
            self.pipeline.reset()
        pf = await self.step()
        if was_running:
            await self.start()
        return pf

    async def update_config(self, patch: dict) -> AppConfig:
        new_cfg = AppConfig.model_validate(deep_merge(self.cfg.model_dump(), patch))  # raises on invalid
        scene_changed = new_cfg.scene != self.cfg.scene or new_cfg.stream.loop_file_sequence != self.cfg.stream.loop_file_sequence
        with self._lock:
            new_source = FrameSource(new_cfg) if scene_changed else self.source
            self.pipeline.configure(new_cfg)
            self.cfg = new_cfg
            self.source = new_source
            self.config_version += 1
        if scene_changed:
            await self.reset()
        elif not self.running:
            pf = await asyncio.to_thread(self._reprocess_current)
            if pf:
                self._publish(pf, count_fps=False)
        return self.cfg

    async def reset_config(self) -> AppConfig:
        return await self.update_config(AppConfig().model_dump())

    def get_frame(self, frame_id: int) -> ProcessedFrame | None:
        return self.history.get(frame_id)

    # ------------------------------------------------------------------ benchmark
    async def run_benchmark(self, frames: int, scaling: bool) -> None:
        if self.benchmark_state["running"]:
            raise RuntimeError("a benchmark is already running")
        self.benchmark_state.update(running=True, progress=0.0, message="starting", error=None)

        def progress(p: float, msg: str) -> None:
            self.benchmark_state.update(progress=p, message=msg)

        cfg = self.cfg.model_copy(deep=True)

        async def job():
            try:
                result = await asyncio.to_thread(
                    run_benchmark, cfg, frames, (16, 32, 64, 96, 128) if scaling else (), 6, progress)
                self.benchmark_state.update(result=result)
            except Exception as exc:
                self.benchmark_state.update(error=f"{type(exc).__name__}: {exc}")
            finally:
                self.benchmark_state.update(running=False)

        asyncio.create_task(job())
