"""FastAPI application: REST API, WebSocket stream and (when built) the dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import __version__
from backend.api import routes, websocket
from backend.core.engine import StreamEngine

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    eng = StreamEngine()
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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"],
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
