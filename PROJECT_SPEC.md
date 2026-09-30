# PS 26053: specification traceability

The prototype was built from *SIH_26053_Adaptive_2_5D_LiDAR_Prototype_Specification.md*. This file maps each acceptance-checklist item (spec §22) and each non-negotiable (spec §23) to where it is implemented and how it is verified.

## Acceptance checklist (spec §22)

| Item | Implemented in | Verified by |
|---|---|---|
| Launches locally with documented commands | `start.ps1`, `start.sh`, `python -m backend`, README §1 | manual |
| Synthetic frame without external downloads | `backend/lidar/synthetic_scene.py` (ray-cast, seeded) | `test_synthetic_scene_is_deterministic`, `test_all_presets_contain_required_structures` |
| Displays point cloud and 2.5D map | `PointCloudView.tsx`, `MapCanvas.tsx`, `ReliefView.tsx` | demo |
| Semantic classes visible in map and legend | `MapCanvas.tsx`, `Dashboard.tsx` legend, `/api/classes` | demo |
| Configurable distance bands | `MapConfig.bands`, `ControlPanel.tsx` band editor, `POST /api/config` | `test_custom_non_power_of_two_schedule`, `test_invalid_schedules_rejected` |
| 5 / 10 / 25 / 50 cm reference bands | `REFERENCE_BANDS` in `core/config.py` | `test_level_for_range_half_open`, `test_cell_size_for_range` |
| Deterministic, tested cell boundaries | `ResolutionSchedule.project` (integer quanta, half-open, tile-centre rule) | `tests/test_boundaries.py` (10 / 30 / 60 / 100 m, both axes) |
| Cells keep elevation + semantic statistics | `mapping/projection.py`, `mapping/cell.py` | `test_vectorised_matches_reference`, `test_elevation_aggregation_keeps_overhang`, `test_mixed_class_cell_keeps_histogram` |
| Dynamic objects across ≥ 2 frames | `tracking/tracker.py`, `core/pipeline.py::_objects` | `test_dynamic_objects_detected_across_frames` |
| Uniform and adaptive use the same input | `core/pipeline.py` (both aggregate identical `x, y, z, labels`) | `test_compare_uses_same_points` |
| Cell count / memory proxy from actual data | `mapping/baseline.py::compare`, `CELL_RECORD_DTYPE` | `test_compare_uses_same_points`, `test_pipeline_metrics_are_measured` |
| Latency and FPS measured and shown live | `core/pipeline.py` stage timers, `core/metrics.py::FpsMeter`, `Header.tsx` | `test_pipeline_metrics_are_measured` |
| UI explains why adaptive reduces load | KPI strip, split view, band overlay, range-profile chart, Architecture tab | demo |
| README: run + plug in a trained model | `README.md` §1, §6 | — |

## Non-negotiables (spec §23)

1. Executable locally: yes. 2. React + Vite + TS / Python + FastAPI: yes. 3. Three.js / R3F for 3D: yes. 4. Mapping engine implemented for real: `mapping/`. 5. Sparse hash map: sorted-key sparse arrays, one record per occupied cell. 6. Configurable reference bands: yes. 7. Keys encode level + integer indices: `pack_cell_key`. 8. Boundary conventions unit-tested at 10 / 30 / 60 / 100 m: yes. 9. Every required cell field: `CellArrays` / `Cell`. 10. Adaptive and uniform on the same frame: yes. 11. Reduction + memory proxy from runtime values: yes. 12. Measured latency / FPS: yes. 13. Deterministic synthetic scenes: yes. 14. Pluggable segmentation with a labelled fallback: `SegmentationModel`, UI badge. 15. README with setup, commands, architecture, API, tests, demo: yes.

## Design decisions beyond the spec

* **Alignment tiles.** The spec's sizes are not a power-of-two hierarchy. Tiles of LCM size, each assigned one level, give an exact partition without forcing incompatible parent/child relations. This is the "explicit boundary handling" option described in spec §8.3.
* **Coordinate origin.** The vehicle frame origin is on the ground under the LiDAR (z = elevation). Sensor-frame files can be shifted with `scene.import_z_offset_m`.
* **2.5D clustering.** Body and overhead layers are clustered separately, so objects under overhangs stay distinct. This mirrors the 2.5D idea of keeping height.
* **Binary WebSocket protocol.** Required to stream ~40–50 k cells at sensor rate. The JSON cell endpoint remains for inspection.
