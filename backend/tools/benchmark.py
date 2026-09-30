"""Command-line benchmark (same code path as the dashboard's "Benchmark report").

    python -m backend.tools.benchmark --frames 100 --preset urban
"""

from __future__ import annotations

import argparse

from backend.core.benchmark import run_benchmark
from backend.core.config import AppConfig, SceneConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--preset", default="urban", choices=["urban", "road", "mixed_traffic", "terrain"])
    ap.add_argument("--beams", type=int, default=64)
    ap.add_argument("--no-scaling", action="store_true")
    args = ap.parse_args()
    cfg = AppConfig(scene=SceneConfig(preset=args.preset, beams=args.beams))

    def progress(p: float, msg: str) -> None:
        print(f"\r[{p * 100:5.1f}%] {msg:<40}", end="", flush=True)

    r = run_benchmark(cfg, args.frames, () if args.no_scaling else (16, 32, 64, 96, 128), 6, progress)
    s = r["summary"]
    print("\n")
    print(f"Frames: {r['frames']}   preset: {args.preset}   beams: {args.beams}   ({r['duration_s']:.1f} s)")
    print(f"{'metric':<38}{'adaptive':>14}{'uniform 5 cm':>16}")
    print(f"{'points / frame (median)':<38}{s['points']['median']:>14,.0f}")
    print(f"{'allocated cells (median)':<38}{s['adaptive_cells']['median']:>14,.0f}{s['uniform_sparse_cells']['median']:>16,.0f}")
    print(f"{'dense cells over 100 m disk':<38}{'':>14}{s['uniform_dense_cells']:>16,.0f}")
    print(f"{'memory proxy MB (median)':<38}{s['adaptive_mb']['median']:>14.2f}{s['uniform_sparse_mb']['median']:>16.2f}")
    print(f"{'dense memory proxy MB':<38}{'':>14}{s['uniform_dense_mb']:>16.1f}")
    print(f"{'projection+aggregation ms (median)':<38}{s['adaptive_projection_ms']['median']:>14.2f}{s['uniform_projection_ms']['median']:>16.2f}")
    print(f"{'end-to-end latency ms (median / p95)':<38}{s['latency_ms']['median']:>8.1f} / {s['latency_ms']['p95']:.1f}")
    print(f"{'reduction vs dense 5 cm':<38}{s['reduction_vs_dense_pct']['median']:>13.2f}%")
    print(f"{'reduction vs sparse 5 cm':<38}{s['reduction_vs_sparse_pct']['median']:>13.2f}%")
    print(f"{'alignment check (all frames)':<38}{'PASS' if s['partition_ok_all'] else 'FAIL':>14}")
    if s.get("seg_accuracy"):
        print(f"{'fallback segmentation acc. vs GT':<38}{100 * s['seg_accuracy']['median']:>13.1f}%")
    if r["scaling"]:
        print("\nDensity sweep (median over frames):")
        print(f"{'beams':>6}{'points':>10}{'latency ms':>12}{'adaptive ms':>13}{'uniform ms':>12}{'adaptive cells':>16}{'uniform cells':>15}")
        for row in r["scaling"]:
            print(f"{row['beams']:>6}{row['points']:>10,.0f}{row['latency_ms']:>12.1f}{row['adaptive_projection_ms']:>13.2f}"
                  f"{row['uniform_projection_ms']:>12.2f}{row['adaptive_cells']:>16,.0f}{row['uniform_sparse_cells']:>15,.0f}")
    print(f"\nSaved: {r.get('saved_to')}")


if __name__ == "__main__":
    main()
