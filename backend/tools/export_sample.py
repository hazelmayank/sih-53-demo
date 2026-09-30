"""Export synthetic frames as a KITTI-style .bin sequence (for exercising the file-import path offline).

    python -m backend.tools.export_sample --preset mixed_traffic --frames 20
"""

from __future__ import annotations

import argparse

from backend.core.config import SceneConfig
from backend.lidar.io import DATA_ROOT, write_kitti_bin
from backend.lidar.synthetic_scene import SyntheticLidar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="mixed_traffic", choices=["urban", "road", "mixed_traffic", "terrain"])
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--beams", type=int, default=64)
    ap.add_argument("--sensor-frame", action="store_true",
                    help="write z relative to the LiDAR (like raw KITTI); load with import_z_offset_m=1.8")
    args = ap.parse_args()
    cfg = SceneConfig(preset=args.preset, seed=args.seed, beams=args.beams)
    src = SyntheticLidar(cfg)
    out = DATA_ROOT / "sample" / f"{args.preset}_seq"
    for i in range(args.frames):
        f = src.frame(i)
        xyz = f.xyz.copy()
        if args.sensor_frame:
            xyz[:, 2] -= cfg.sensor_height_m
        write_kitti_bin(out / f"{i:06d}.bin", xyz, f.intensity)
    print(f"wrote {args.frames} frames to {out}")


if __name__ == "__main__":
    main()
