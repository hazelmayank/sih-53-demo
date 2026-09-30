"""Small end-to-end frames: synthetic generator determinism, segmentation, tracking, API."""

import json
import struct

import numpy as np
import pytest

from backend.core.config import NUM_CLASSES, AppConfig, SceneConfig, SemanticClass as C
from backend.core.pipeline import PerceptionPipeline
from backend.lidar.frame import validate_points
from backend.lidar.io import read_points, write_kitti_bin
from backend.lidar.synthetic_scene import SyntheticLidar


@pytest.fixture(scope="module")
def urban_run():
    cfg = AppConfig(scene=SceneConfig(preset="urban", beams=48))
    pipe = PerceptionPipeline(cfg)
    src = SyntheticLidar(cfg.scene)
    frames = [pipe.process(src.frame(i)) for i in range(8)]
    return cfg, frames


def test_synthetic_scene_is_deterministic():
    cfg = SceneConfig(preset="mixed_traffic", beams=32, seed=3)
    a = SyntheticLidar(cfg).frame(4)
    b = SyntheticLidar(cfg).frame(4)
    assert np.array_equal(a.xyz, b.xyz) and np.array_equal(a.gt_label, b.gt_label)
    c = SyntheticLidar(cfg).frame(5)
    assert not np.array_equal(a.xyz[:100], c.xyz[:100])  # noise and actors change per frame


@pytest.mark.parametrize("preset", ["urban", "road", "mixed_traffic", "terrain"])
def test_all_presets_contain_required_structures(preset):
    f = SyntheticLidar(SceneConfig(preset=preset, beams=32)).frame(0)
    labels = set(np.unique(f.gt_label).tolist())
    assert {C.DRIVABLE_TERRAIN, C.NON_DRIVABLE_TERRAIN, C.STATIC_OBSTACLE, C.VEHICLE, C.PEDESTRIAN} <= labels
    r = np.hypot(f.xyz[:, 0], f.xyz[:, 1])
    assert r.max() > 60  # long-range returns exist


def test_labels_in_class_set(urban_run):
    _, frames = urban_run
    pf = frames[-1]
    assert sum(s["points"] for s in pf.class_stats) == pf.metrics["points"]
    assert set(np.unique(pf.cells.dominant_label).tolist()) <= set(range(NUM_CLASSES))


def test_fallback_segmentation_quality(urban_run):
    _, frames = urban_run
    ev = frames[-1].segmentation_eval
    assert ev is not None and ev["accuracy"] > 0.85
    assert ev["iou"][C.DRIVABLE_TERRAIN] > 0.85


def test_dynamic_objects_detected_across_frames(urban_run):
    _, frames = urban_run
    pf = frames[-1]
    moving = [o for o in pf.objects if o["dynamic"]]
    assert moving, "at least one moving object should be tracked after 8 frames"
    assert all(o["track_hits"] >= 3 for o in moving)
    assert all(o["class"] != C.STATIC_OBSTACLE for o in moving)
    # stable IDs: the fastest moving track exists in the previous frame too
    tid = max(moving, key=lambda o: o["speed"])["track_id"]
    assert any(o["track_id"] == tid for o in frames[-2].objects)


def test_partition_holds_on_real_frames(urban_run):
    _, frames = urban_run
    assert all(pf.partition["ok"] for pf in frames)


def test_malformed_points_rejected():
    with pytest.raises(ValueError):
        validate_points(np.zeros((10, 2)))
    with pytest.raises(ValueError):
        validate_points(np.zeros((10, 3)), intensity=np.zeros(9))
    xyz, it, ts = validate_points(np.array([[1, 2, 3], [4, 5, 6]]), intensity=np.array([10, 255]))
    assert it.max() == pytest.approx(1.0)  # 0-255 reflectivity is normalised


def test_nan_points_are_dropped():
    cfg = AppConfig(scene=SceneConfig(preset="road", beams=16))
    raw = SyntheticLidar(cfg.scene).frame(0)
    raw.xyz[:50] = np.nan
    pf = PerceptionPipeline(cfg).process(raw)
    assert pf.preprocess["non_finite"] == 50


def test_kitti_bin_roundtrip(tmp_path):
    xyz = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    it = np.linspace(0, 1, 500, dtype=np.float32)
    p = tmp_path / "000000.bin"
    write_kitti_bin(p, xyz, it)
    xyz2, it2 = read_points(p)
    assert np.allclose(xyz, xyz2) and np.allclose(it, it2)


def test_api_and_binary_stream():
    from fastapi.testclient import TestClient

    from backend.app import app

    with TestClient(app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        assert len(c.get("/api/classes").json()) == NUM_CLASSES
        bad = c.post("/api/config", json={"map": {"bands": [{"upper_m": 10, "cell_size_m": 0.37}, {"upper_m": 100, "cell_size_m": 0.41}]}})
        assert bad.status_code == 422
        with c.websocket_connect("/ws/stream") as ws:
            ws.send_text(json.dumps({"type": "subscribe", "points": False, "uniform": True}))
            buf = ws.receive_bytes()
        hlen = struct.unpack("<I", buf[:4])[0]
        header = json.loads(buf[4:4 + hlen])
        base = (4 + hlen + 7) // 8 * 8
        sec = {s["name"]: s for s in header["sections"]}
        lv = np.frombuffer(buf, np.uint8, sec["c_level"]["length"], base + sec["c_level"]["offset"])
        assert lv.size == header["counts"]["cells"] == header["metrics"]["occupied_cells"]
        c.post("/api/scene/stop")
        fid = header["frame_id"]
        assert c.get(f"/api/frame/{fid}").status_code == 200
