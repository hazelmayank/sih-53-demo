"""REST endpoints."""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Body, File, HTTPException, Query, Request, UploadFile
from pydantic import ValidationError

from backend import __version__
from backend.core.config import CLASS_INFO, AppConfig
from backend.core.engine import StreamEngine
from backend.core.models import (
    BenchmarkRunRequest,
    CellDetail,
    ClassInfo,
    HealthResponse,
    SceneStartRequest,
    SeekRequest,
    StatusResponse,
)
from backend.lidar.io import DATA_ROOT, SUPPORTED, list_sequences, read_points
from backend.lidar.synthetic_scene import PRESETS

router = APIRouter(prefix="/api")


def engine(request: Request) -> StreamEngine:
    return request.app.state.engine


def _bad_config(exc: Exception) -> HTTPException:
    if isinstance(exc, ValidationError):
        msgs = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" if e["loc"] else e["msg"] for e in exc.errors()]
        return HTTPException(status_code=422, detail="; ".join(msgs))
    return HTTPException(status_code=422, detail=str(exc))


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    st = engine(request).status()
    return HealthResponse(status="ok" if not st["error"] else "degraded", version=__version__,
                          uptime_s=st["uptime_s"], running=st["running"], frame_index=st["frame_index"],
                          segmenter=st["segmenter"], rss_mb=st["rss_mb"])


@router.get("/config", response_model=AppConfig)
async def get_config(request: Request):
    return engine(request).cfg


@router.post("/config", response_model=AppConfig)
async def post_config(request: Request, patch: dict[str, Any] = Body(...)):
    """Partial update, e.g. ``{"map": {"bands": [...]}, "processing": {"max_range_m": 80}}``."""
    try:
        return await engine(request).update_config(patch)
    except (ValidationError, ValueError, RuntimeError) as exc:
        raise _bad_config(exc) from exc


@router.post("/config/reset", response_model=AppConfig)
async def reset_config(request: Request):
    return await engine(request).reset_config()


@router.get("/status", response_model=StatusResponse)
async def status(request: Request):
    return StatusResponse(status=engine(request).status())


@router.post("/scene/start", response_model=StatusResponse)
async def scene_start(request: Request, body: SceneStartRequest | None = None):
    eng = engine(request)
    if body:
        patch = {k: v for k, v in body.model_dump().items() if v is not None}
        if patch:
            try:
                await eng.update_config({"scene": patch})
            except (ValidationError, ValueError, RuntimeError) as exc:
                raise _bad_config(exc) from exc
    await eng.start()
    return StatusResponse(status=eng.status())


@router.post("/scene/stop", response_model=StatusResponse)
async def scene_stop(request: Request):
    eng = engine(request)
    eng.auto_paused = False  # an explicit pause survives viewers reconnecting
    await eng.stop()
    return StatusResponse(status=eng.status())


@router.post("/scene/step", response_model=StatusResponse)
async def scene_step(request: Request):
    eng = engine(request)
    await eng.step()
    return StatusResponse(status=eng.status())


@router.post("/scene/reset", response_model=StatusResponse)
async def scene_reset(request: Request):
    eng = engine(request)
    await eng.reset()
    return StatusResponse(status=eng.status())


@router.post("/scene/seek", response_model=StatusResponse)
async def scene_seek(request: Request, body: SeekRequest):
    eng = engine(request)
    await eng.seek(body.frame_index)
    return StatusResponse(status=eng.status())


@router.get("/scene/presets")
async def scene_presets():
    return PRESETS


@router.get("/scene/files")
async def scene_files():
    return list_sequences()


@router.post("/scene/upload")
async def scene_upload(request: Request, file: UploadFile = File(...)):
    """Upload a single point-cloud file into data/uploads/ and switch the source to it."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", file.filename or "upload.bin")
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in SUPPORTED:
        raise HTTPException(400, f"unsupported extension {ext!r}; use one of {sorted(SUPPORTED)}")
    dest = DATA_ROOT / "uploads" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    if len(data) > 200 * 2**20:
        raise HTTPException(413, "file too large (200 MB limit)")
    dest.write_bytes(data)
    try:
        xyz, _ = read_points(dest)
        if len(xyz) == 0:
            raise ValueError("file contains no points")
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"could not parse point cloud: {exc}") from exc
    try:
        await engine(request).update_config({"scene": {"source": "file", "file_sequence": f"uploads/{name}"}})
    except (ValidationError, ValueError, RuntimeError) as exc:
        raise _bad_config(exc) from exc
    return {"ok": True, "path": f"uploads/{name}", "points": int(len(xyz))}


@router.get("/frame/{frame_id}")
async def frame_summary(request: Request, frame_id: int):
    pf = engine(request).get_frame(frame_id)
    if pf is None:
        raise HTTPException(404, f"frame {frame_id} is not in the history buffer")
    from backend.api.protocol import _sanitize

    return _sanitize(pf.summary())


@router.get("/frame/{frame_id}/cells")
async def frame_cells(request: Request, frame_id: int, limit: int = Query(2000, ge=1, le=200000),
                      level: int | None = None):
    """Full JSON cell records (for inspection / export; the dashboard uses the binary stream)."""
    pf = engine(request).get_frame(frame_id)
    if pf is None:
        raise HTTPException(404, f"frame {frame_id} is not in the history buffer")
    c = pf.cells
    idx = range(len(c)) if level is None else [i for i in range(len(c)) if int(c.level[i]) == level]
    out = []
    for i in idx:
        if len(out) >= limit:
            break
        out.append(c.cell(i).to_dict())
    return {"frame_id": frame_id, "total_cells": len(c), "returned": len(out), "cells": out}


@router.get("/frame/{frame_id}/cell", response_model=CellDetail)
async def frame_cell(request: Request, frame_id: int, level: int, ix: int, iy: int):
    pf = engine(request).get_frame(frame_id)
    if pf is None:
        raise HTTPException(404, f"frame {frame_id} is not in the history buffer")
    i = pf.cells.find(level, ix, iy)
    if i is None:
        raise HTTPException(404, "cell not allocated (no points fell in it)")
    cell = pf.cells.cell(i)
    b = pf.schedule.bands[cell.level]
    s = cell.cell_size
    d = cell.to_dict()
    return CellDetail(frame_id=frame_id, bounds=[cell.ix * s, cell.iy * s, (cell.ix + 1) * s, (cell.iy + 1) * s],
                      band={"level": b.level, "lower_m": b.lower_m, "upper_m": b.upper_m, "cell_size_m": b.cell_size_m},
                      **d)


@router.get("/benchmark")
async def benchmark(request: Request, series: int = Query(200, ge=0, le=600)):
    eng = engine(request)
    latest = eng.latest
    return {
        "live": {
            "summary": eng.metrics.summary(),
            "series": eng.metrics.series(series),
            "latest": latest.summary()["comparison"] if latest else None,
        },
        "run": {k: v for k, v in eng.benchmark_state.items()},
    }


@router.post("/benchmark/run")
async def benchmark_run(request: Request, body: BenchmarkRunRequest | None = None):
    body = body or BenchmarkRunRequest()
    try:
        await engine(request).run_benchmark(body.frames, body.scaling)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@router.get("/classes", response_model=list[ClassInfo])
async def classes():
    return CLASS_INFO
