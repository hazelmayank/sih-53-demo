"""Pydantic request / response models for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.core.config import ScenePreset


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_s: float
    running: bool
    frame_index: int
    segmenter: dict[str, Any]
    rss_mb: float | None = None


class ClassInfo(BaseModel):
    id: int
    name: str
    label: str
    color: str
    description: str


class SceneStartRequest(BaseModel):
    preset: ScenePreset | None = None
    source: str | None = Field(default=None, pattern="^(synthetic|file)$")
    file_sequence: str | None = None
    seed: int | None = Field(default=None, ge=0)


class SeekRequest(BaseModel):
    frame_index: int = Field(ge=0)


class BenchmarkRunRequest(BaseModel):
    frames: int = Field(default=100, ge=5, le=500)
    scaling: bool = True


class CellDetail(BaseModel):
    frame_id: int
    level: int
    cell_size: float
    ix: int
    iy: int
    center_x: float
    center_y: float
    range_m: float
    bounds: list[float]
    point_count: int
    min_z: float
    max_z: float
    mean_z: float
    roughness: float
    dominant_label: int
    class_histogram: dict[str, int]
    occupancy: float
    dynamic_score: float
    confidence: float
    last_seen_frame: int
    band: dict[str, Any]


class StatusResponse(BaseModel):
    ok: bool = True
    status: dict[str, Any]
