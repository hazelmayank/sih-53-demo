"""Runtime configuration.

Coordinate convention (used everywhere, backend and frontend):
    vehicle frame with the origin on the ground directly below the LiDAR,
    x = forward, y = left, z = up, metres. The LiDAR sits at (0, 0, sensor_height_m),
    so z of a map cell is its elevation above the local ground plane at the vehicle.
    Radial (planar) range r = sqrt(x^2 + y^2) is measured from the sensor axis.
    Imported sensor-frame files (origin at the LiDAR, e.g. KITTI) are shifted up by
    ``import_z_offset_m`` on load.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SemanticClass(IntEnum):
    DRIVABLE_TERRAIN = 0
    NON_DRIVABLE_TERRAIN = 1
    STATIC_OBSTACLE = 2
    VEHICLE = 3
    PEDESTRIAN = 4
    UNKNOWN = 5


NUM_CLASSES = len(SemanticClass)

CLASS_INFO: list[dict[str, Any]] = [
    {"id": 0, "name": "DRIVABLE_TERRAIN", "label": "Drivable terrain", "color": "#2bb39a",
     "description": "Road or traversable surface."},
    {"id": 1, "name": "NON_DRIVABLE_TERRAIN", "label": "Non-drivable terrain", "color": "#b0803f",
     "description": "Grass, rough ground, ditch, curb, pothole, steep surface."},
    {"id": 2, "name": "STATIC_OBSTACLE", "label": "Static obstacle", "color": "#f2a93b",
     "description": "Wall, barrier, pole, building edge, overhanging structure."},
    {"id": 3, "name": "VEHICLE", "label": "Vehicle", "color": "#4ea8ff",
     "description": "Car, bus, truck or other road vehicle."},
    {"id": 4, "name": "PEDESTRIAN", "label": "Pedestrian", "color": "#ff4f8b",
     "description": "Person or person-like dynamic object."},
    {"id": 5, "name": "UNKNOWN", "label": "Unknown", "color": "#8b95a5",
     "description": "Low-confidence / unclassified point."},
]

TERRAIN_CLASSES = (SemanticClass.DRIVABLE_TERRAIN, SemanticClass.NON_DRIVABLE_TERRAIN)

# Reference schedule from the problem statement: [lower, upper) -> cell size.
REFERENCE_BANDS: list[dict[str, float]] = [
    {"upper_m": 10.0, "cell_size_m": 0.05},
    {"upper_m": 30.0, "cell_size_m": 0.10},
    {"upper_m": 60.0, "cell_size_m": 0.25},
    {"upper_m": 100.0, "cell_size_m": 0.50},
]

ScenePreset = Literal["road", "urban", "mixed_traffic", "terrain"]


class BandConfig(BaseModel):
    upper_m: float = Field(gt=0, le=250, description="Exclusive upper range of the band (m).")
    cell_size_m: float = Field(ge=0.01, le=5.0, description="Cell edge length inside the band (m).")


class SceneConfig(BaseModel):
    source: Literal["synthetic", "file"] = "synthetic"
    preset: ScenePreset = "urban"
    seed: int = Field(default=7, ge=0, le=2**31 - 1)
    beams: int = Field(default=64, ge=8, le=128)
    azimuth_resolution_deg: float = Field(default=0.2, ge=0.08, le=1.0)
    sensor_height_m: float = Field(default=1.8, ge=0.5, le=4.0)
    range_noise_m: float = Field(default=0.015, ge=0.0, le=0.3)
    dropout: float = Field(default=0.02, ge=0.0, le=0.8)
    outlier_ratio: float = Field(default=0.002, ge=0.0, le=0.05)
    file_sequence: str | None = Field(default=None, description="Folder or file under data/ for source=file.")
    import_z_offset_m: float = Field(default=0.0, ge=-5.0, le=5.0,
                                     description="Added to z of imported files (1.73 for raw KITTI velodyne).")


class ProcessingConfig(BaseModel):
    min_range_m: float = Field(default=1.5, ge=0.0, le=10.0)
    max_range_m: float = Field(default=100.0, gt=5.0, le=200.0)
    outlier_filter: bool = True
    voxel_downsample_m: float = Field(default=0.0, ge=0.0, le=1.0, description="0 disables downsampling.")
    ground_sectors: int = Field(default=360, ge=36, le=1440)
    ground_max_slope: float = Field(default=0.4, ge=0.05, le=1.5)
    obstacle_height_m: float = Field(default=0.25, ge=0.05, le=1.0)
    cluster_cell_m: float = Field(default=0.4, ge=0.1, le=2.0)
    min_cluster_points: int = Field(default=4, ge=1, le=100)
    segmenter: Literal["geometric_fallback", "oracle_gt", "torchscript"] = "geometric_fallback"
    model_path: str | None = None
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    motion_threshold_mps: float = Field(default=0.7, ge=0.05, le=10.0)
    track_gate_m: float = Field(default=2.5, ge=0.2, le=10.0)
    track_max_misses: int = Field(default=3, ge=0, le=30)


class MapConfig(BaseModel):
    bands: list[BandConfig] = Field(default_factory=lambda: [BandConfig(**b) for b in REFERENCE_BANDS])
    uniform_cell_m: float = Field(default=0.05, ge=0.01, le=5.0)

    @field_validator("bands")
    @classmethod
    def _bands_sorted(cls, bands: list[BandConfig]) -> list[BandConfig]:
        if not bands:
            raise ValueError("at least one resolution band is required")
        if len(bands) > 8:
            raise ValueError("at most 8 resolution bands are supported")
        uppers = [b.upper_m for b in bands]
        if any(b <= a for a, b in zip(uppers, uppers[1:])):
            raise ValueError("band upper bounds must be strictly increasing")
        return bands


class StreamConfig(BaseModel):
    target_hz: float = Field(default=10.0, ge=0.5, le=30.0)
    playback_speed: float = Field(default=1.0, ge=0.1, le=4.0)
    max_stream_points: int = Field(default=60000, ge=0, le=300000)
    history_frames: int = Field(default=30, ge=2, le=200)
    loop_file_sequence: bool = True


class AppConfig(BaseModel):
    scene: SceneConfig = Field(default_factory=SceneConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    map: MapConfig = Field(default_factory=MapConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)

    @model_validator(mode="after")
    def _check_schedule(self) -> "AppConfig":
        # Imported lazily to avoid a circular import; raises ValueError with a readable message.
        from backend.mapping.adaptive_grid import ResolutionSchedule

        ResolutionSchedule.from_config(self.map, self.processing.max_range_m)
        ResolutionSchedule.uniform(self.map.uniform_cell_m, self.processing.max_range_m)
        return self


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``patch`` into a copy of ``base`` (lists are replaced, not merged)."""
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
