"""FastAPI application: REST API, WebSocket stream and (when built) the dashboard."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import __version__
from backend.api import routes, websocket
from backend.core.config import AppConfig, deep_merge
from backend.core.engine import StreamEngine

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


def config_from_env() -> AppConfig:
    """Optional overrides for hosted deployments (e.g. a small cloud instance).

    LIDAR_BEAMS, LIDAR_TARGET_HZ, LIDAR_HISTORY_FRAMES, LIDAR_MAX_STREAM_POINTS, LIDAR_PRESET
    """
    env = os.environ
    patch: dict = {"scene": {}, "stream": {}}
    if "LIDAR_BEAMS" in env:
        patch["scene"]["beams"] = int(env["LIDAR_BEAMS"])
    if "LIDAR_PRESET" in env:
        patch["scene"]["preset"] = env["LIDAR_PRESET"]
    if "LIDAR_TARGET_HZ" in env:
        patch["stream"]["target_hz"] = float(env["LIDAR_TARGET_HZ"])
    if "LIDAR_HISTORY_FRAMES" in env:
        patch["stream"]["history_frames"] = int(env["LIDAR_HISTORY_FRAMES"])
    if "LIDAR_MAX_STREAM_POINTS" in env:
        patch["stream"]["max_stream_points"] = int(env["LIDAR_MAX_STREAM_POINTS"])
    return AppConfig.model_validate(deep_merge(AppConfig().model_dump(), patch))


@asynccontextmanager
async def lifespan(app: FastAPI):
    eng = StreamEngine(config_from_env())
    app.state.engine = eng
    await eng.step()  # frame 0 is ready before the first client connects
    yield
    await eng.stop()


app = FastAPI(
    title="Adaptive 2.5D LiDAR Mapping (SIH PS 26053)",
    version=__version__,
    description="Foveated, variable-resolution 2.5D semantic mapping of LiDAR point clouds.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173",
                   *[o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes.router)
app.include_router(websocket.router)

if FRONTEND_DIST.exists():
    # Production / offline demo: serve the built dashboard from the same origin.
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        f = FRONTEND_DIST / path
        if path and f.is_file() and FRONTEND_DIST in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(FRONTEND_DIST / "index.html")
