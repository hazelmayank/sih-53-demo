# Adaptive Variable-Resolution 2.5D LiDAR Mapping
**SIH 2025 · Problem Statement 26053 · DRDO · Smart Vehicles**

A working prototype of a *foveated* LiDAR perception and mapping pipeline. Point clouds are classified into semantic classes and projected into a sparse 2.5D elevation map. Cells are **5 cm near the vehicle and grow to 50 cm by 100 m**. They keep min/max/mean elevation, roughness and a full class histogram, so height and semantics survive the coarsening. A live dashboard shows every stage and compares the adaptive map with a uniform 5 cm grid built from the same points and labels.

Every number the dashboard shows is measured at runtime. Nothing is hard-coded.

---

## 1. Quick start

Requirements: Python 3.10+ and Node.js 18+ (Node is only needed once, to build the dashboard).

```powershell
# Windows, one command (installs deps and builds the dashboard on first run)
.\start.ps1
```
```bash
# Linux / macOS / Git Bash
./start.sh
```
Then open **http://127.0.0.1:8053**. Streaming starts as soon as the dashboard connects.

### Manual setup
```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python -m backend --port 8053          # API + WebSocket + built dashboard
```

### Development mode (hot reload)
```bash
python -m uvicorn backend.app:app --port 8053 --reload      # terminal 1
cd frontend && npm run dev                                   # terminal 2 -> http://localhost:5173
```
Vite proxies `/api` and `/ws` to `127.0.0.1:8053`. Port 8053 was chosen to avoid clashing with other local services on 8000.

**Offline:** once `pip install` and `npm run build` have run, the demo needs no internet. There are no CDNs, no web fonts and no dataset downloads. The synthetic LiDAR is generated locally.

### Docker
```bash
docker compose up --build      # -> http://localhost:8053
```

### Hosted link on Render (dashboard + API + live stream on one URL)
1. Render dashboard → **New → Blueprint** → select this GitHub repo. `render.yaml` configures a Docker web service.
2. Wait for the build (~5 min). Open `https://<service-name>.onrender.com`.

The service listens on Render's `$PORT`. Environment variables set lighter defaults for a small CPU: `LIDAR_BEAMS=32`, `LIDAR_TARGET_HZ=5`, `LIDAR_HISTORY_FRAMES=10`, `LIDAR_MAX_STREAM_POINTS=30000`. The stream pauses automatically when the last viewer leaves and resumes when someone opens the page. On the free plan (0.1 CPU) expect roughly 1 frame/s, and a ~1 min cold start after 15 min idle. The *starter* plan is much smoother. The latency/FPS shown online are the cloud machine's numbers; demo live from a laptop for representative performance.

---

## 2. What the prototype demonstrates

| Capability | Where to see it |
|---|---|
| Point-cloud ingestion: deterministic ray-cast synthetic LiDAR, or imported `.bin/.pcd/.ply/.npy/.csv` | *Point cloud (3D)* tab, frame counter, **Load file** button |
| Terrain analysis (drivable / non-drivable / unknown) | Semantic legend, map colours, IoU vs ground truth |
| Obstacles: walls, poles, vehicles, pedestrians, overhangs | Object boxes on the map, object list |
| 2.5D map: elevation + semantics per cell | *2.5D relief (3D)* tab, cell inspector |
| Adaptive resolution 5 → 10 → 25 → 50 cm | Band rings, **Colour by → Resolution level**, zoom buttons |
| Alignment safety (no gaps, no overlaps) | **Grid debug** toggle, *Alignment check PASS* card, unit tests |
| Performance: latency, FPS, cells, memory | Header, KPI strip, bottom charts, **Benchmark report** |
| Uniform 5 cm baseline on the same frame | *Adaptive vs uniform* split view, charts |
| Dynamic objects across frames | Velocity vectors, trails, *Moving* filter in the object list |

---

## 3. Architecture

```
LiDAR input ─► Preprocess ─► Ground ─► 2.5D clustering ─► Segmentation ─► Objects + tracking
 (sim / file)   range gate    polar      body/overhead      pluggable model    NN tracker, gating
                outliers      line       layers                                 motion test
                                                                   │
                                   ┌───────────────────────────────┘
                                   ▼
                     Adaptive projection + 2.5D aggregation ──► Metrics ──► WebSocket (binary) ──► React dashboard
                     Uniform 5 cm baseline (same points/labels) ─┘              REST API
```

| Module | Responsibility |
|---|---|
| `backend/lidar/synthetic_scene.py` | Deterministic ray-cast spinning LiDAR (64 beams × 0.2°). Four presets: urban, road, mixed traffic, off-road terrain. Moving actors, ground truth for evaluation. |
| `backend/lidar/io.py` | Pure-NumPy readers for KITTI `.bin`, `.pcd`, `.ply`, `.npy`, `.csv`; directory = frame sequence |
| `backend/lidar/preprocess.py` | Malformed-data handling, range gate, isolated-return filter, optional voxel downsampling, polar line-tracking ground model |
| `backend/lidar/clustering.py` | 2.5D connected components: separate body / overhead layers, joined only where a structure is vertically continuous |
| `backend/lidar/segmentation.py` | `SegmentationModel` interface; rule-based fallback; TorchScript adapter; ground-truth oracle (debug) |
| `backend/tracking/tracker.py` | Gated nearest-neighbour association, persistent IDs, smoothed velocity, net-displacement motion test |
| `backend/mapping/adaptive_grid.py` | **Core:** resolution schedule, alignment tiles, `CellKey(level, ix, iy)` |
| `backend/mapping/projection.py` | Vectorised sparse aggregation and runtime partition check |
| `backend/mapping/cell.py` | 2.5D cell record and a reference (per-point loop) aggregator used by the tests |
| `backend/mapping/baseline.py` | Uniform 5 cm baseline, exact dense cell count, comparison |
| `backend/core/pipeline.py` · `engine.py` · `metrics.py` · `benchmark.py` | Orchestration, playback and history, rolling metrics, offline benchmark |
| `backend/api/` | REST routes, WebSocket stream, binary frame protocol |
| `frontend/src/` | React + Vite + TypeScript dashboard (Canvas 2D map, React Three Fiber 3D views, Recharts) |

**Coordinate convention:** vehicle frame with its origin on the ground below the LiDAR; x forward, y left, z up (metres). The sensor is at `(0, 0, 1.8)`, and planar range is `r = √(x² + y²)`. All views use this convention. On the 2D map, forward points up the screen.

---

## 4. The adaptive grid and why it cannot misalign

The reference sizes (5, 10, 25, 50 cm) do not form a power-of-two hierarchy, because 25 cm is 2.5× 10 cm. A plain quadtree therefore cannot nest them. The engine uses **alignment tiles** instead:

1. All cell sizes are whole multiples of a *quantum* (their GCD, 5 cm) and divide a common *alignment tile* (their LCM, 0.5 m).
2. Every level's grid is anchored at the sensor origin. A 0.5 m tile is therefore covered exactly by 10×10 cells of 5 cm, 5×5 of 10 cm, 2×2 of 25 cm or 1×1 of 50 cm.
3. **Each tile gets exactly one level**, chosen from the range of the tile centre with half-open intervals `[lower, upper)`. All points in the tile use that level.
4. Keys are `CellKey = (level, ix, iy)`, packed into one int64. `ix = floor(x / size)` is computed in integer quanta, so a value like `10.0 / 0.05` can never round the wrong way.

As a result, the band boundary is a 0.5 m staircase approximating the circle. The allocated cells form an **exact partition of the plane**: no point is stored twice, no cells overlap and there are no gaps at transitions. `check_partition()` verifies this on every frame (the *Alignment check* card), and `tests/test_grid.py` / `tests/test_boundaries.py` test it at 10, 30, 60 and 100 m.

Range acceptance uses the point's own range with an inclusive outer bound, so a point at exactly 100 m is kept and one at 100.001 m is rejected. The bands are fully configurable from the dashboard. Any sizes work as long as their alignment tile is ≤ 5 m (e.g. 4/12/30 cm gives a 0.6 m tile). Invalid schedules are rejected with a clear message.

### 2.5D cell record
`level, cell_size, ix, iy, center_x/y, point_count, min_z, max_z, mean_z, roughness (σz), dominant_label, class_histogram[6], occupancy, dynamic_score, confidence, last_seen_frame`

min/max keep overhangs and obstacles from being averaged away, and the histogram keeps minority classes in coarse far cells. Occupancy = `1 − exp(−count/2)` separates sparse far cells from empty space. Confidence = histogram purity × mean point confidence.

### Memory figures (labelled honestly)
* **Memory proxy:** active cells × 74 B (a packed 58 B record plus a 16 B sparse-index entry). A dense grid is charged 58 B per cell and no index. This is an estimate, not measured DRAM.
* **Measured cell arrays:** the actual bytes of the NumPy arrays holding the frame's map.
* **Process RSS:** OS-measured resident memory of the whole backend (via `psutil`).

---

## 5. Measured results (this machine, `python -m backend.tools.benchmark`)

100 frames, urban preset, 64 beams (≈ 110 k points per frame), reference bands, measured on the development laptop (AMD Ryzen 7 7735HS, Windows, NumPy on CPU):

| Median per frame | Adaptive | Uniform 5 cm |
|---|---|---|
| Cells if the 100 m disk is allocated densely | 593,592 | **12,566,400** |
| Cells actually allocated (sparse hash map) | **39,388** | 47,652 |
| Reduction vs dense 5 cm | **99.69 %** | — |
| Reduction vs sparse 5 cm | 17.3 % | — |
| Memory proxy | **2.78 MB** | 3.36 MB sparse / **695 MB dense** |
| Projection + aggregation time | 28.0 ms | 28.6 ms |
| Alignment check | PASS on all 100 frames | — |

End-to-end latency, from frame in to adaptive map ready: **137 ms median / 170 ms p95** at 110 k points. Density sweep (6 frames each): 27 k points → 37 ms, 55 k → 53 ms, 111 k → 92 ms, 166 k → 162 ms, 221 k → 210 ms. The adaptive map grows from 11 k to 75 k cells; uniform grows from 14 k to 89 k. Timings on this laptop vary by ±40 % with thermal state, so **run the benchmark on the demo machine** and quote its numbers. The dashboard's *Benchmark report* (or `python -m backend.tools.benchmark`) runs the same 100 frames + sweep and saves JSON to `data/benchmarks/`.

How to read these numbers honestly:
* The large saving is against a fixed-resolution **dense** grid, which is what a naive high-resolution map must allocate and process. Against a *sparse* 5 cm hash map the saving comes only from the 10–100 m bands (the near field is 5 cm in both). It is about 17 % per single scan and grows as frames accumulate or point density rises.
* Projecting and aggregating the points costs the **same** in both representations here. Both are O(N log N) over the same N points. The benefit is in memory and in everything downstream that works per cell (planning, fusion, transmission).

Fallback classifier quality against synthetic ground truth, averaged over 20 frames: point accuracy 93–96 %; mIoU 0.82 urban, 0.76 road, 0.86 mixed traffic, 0.63 off-road terrain; drivable-terrain IoU 0.83–0.96.

---

## 6. Semantic segmentation: fallback and how to plug in a trained model

The fallback is **rule-based and deterministic**, and the UI labels it as such everywhere. It combines height above the estimated ground; local roughness, step and slope; reflectivity bands (asphalt / paint / concrete / vegetation); and cluster shape. Shape covers extent, height, body fill below 1.4 m, whether the object stands on drivable terrain, and whether its top leaves the sensor's field of view. Synthetic ground truth is used **only to score it**, never as input. The optional `oracle_gt` mode, marked "debug" in the UI, feeds ground-truth labels to show the mapping engine independently of classification.

Any network plugs in through one method:

```python
from backend.lidar.segmentation import SegmentationModel, SegmentationResult

class PointNet2Segmenter(SegmentationModel):
    name, kind = "pointnet2", "neural"
    def predict(self, frame, ctx) -> SegmentationResult:
        # frame.xyz (N,3), frame.intensity (N,); ctx has height-above-ground, clusters, ground model
        logits = self.net(frame.xyz, frame.intensity)            # (N, K)
        return SegmentationResult(labels=class_map[logits.argmax(1)], confidence=softmax_max)
```
Register it in `build_segmenter()`. Alternatively, with no code at all, export the trained PointNet++ / KPConv / MinkowskiNet to TorchScript with the contract `forward(FloatTensor[B,N,4] xyz+intensity) -> logits[B,N,K]`, then set:
```json
POST /api/config  {"processing": {"segmenter": "torchscript", "model_path": "models/pn2.pt"}}
```
`TorchScriptSegmenter` batches inference on GPU when available and maps model classes (e.g. SemanticKITTI's 19) onto the 6 map classes through `class_map`. The adaptive grid, baseline, tracker, metrics and dashboard consume only per-point `labels` + `confidence`, so **nothing downstream changes**.

---

## 7. API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Health, version, uptime, segmenter, RSS |
| `/api/config` | GET / POST | Full config / partial update (bands, max range, thresholds, scene…). Invalid values → 422 with reason |
| `/api/config/reset` | POST | Restore defaults |
| `/api/scene/start` · `/stop` · `/step` · `/reset` · `/seek` | POST | Playback control |
| `/api/scene/presets` · `/api/scene/files` | GET | Synthetic presets / importable sequences under `data/` |
| `/api/scene/upload` | POST | Upload a point-cloud file and switch to it |
| `/api/frame/{id}` | GET | Processed-frame summary (metrics, comparison, objects, stage timings) |
| `/api/frame/{id}/cells?limit=&level=` | GET | Full JSON cell records |
| `/api/frame/{id}/cell?level=&ix=&iy=` | GET | One cell incl. class histogram |
| `/api/benchmark` · `/api/benchmark/run` | GET / POST | Live rolling stats; offline 100-frame benchmark + density sweep |
| `/api/classes` | GET | Semantic class definitions and colours |
| `/ws/stream` | WS | Binary processed frames |

Interactive OpenAPI docs are at **http://127.0.0.1:8053/docs**.

**WebSocket frame** (`backend/api/protocol.py`): `uint32 header_len | JSON header | typed-array sections`. The header holds `frame_id, timestamp, sensor, metrics{latency_ms, fps, points, occupied_cells, memory_proxy_mb, …}, objects, comparison, schedule, stage_ms`. Sections hold per-cell arrays (`c_level, c_label, c_ix, c_iy, c_count, c_zmin/zmax/zmean [cm], c_rough, c_conf, c_dyn, c_occ`) plus optional sampled points and uniform-baseline cells. A 40 k-cell map is ≈ 0.75 MB per frame, versus tens of MB as JSON objects. Client messages: `{"type":"subscribe","points":true,"uniform":false}` and `{"type":"get_frame","frame_id":42}`.

---

## 8. Tests

```bash
python -m pytest -q
```
60 tests cover:
* **Boundaries:** r = 9.999 / 10 / 29.999 / 30 / 59.999 / 60 / 100 m on both axes; exactly 100 m accepted, beyond rejected; determinism; float edge cases.
* **Grid:** CellKey round-trip, no collisions across levels, every point inside its cell, partition with no overlaps, exact dense counts, non-power-of-two schedules, invalid schedules rejected.
* **Aggregation:** vectorised engine equals the per-point reference; no point duplicated; min/max keep an overhang; mixed-class histograms; UNKNOWN representable.
* **Benchmark:** the comparison uses identical points, the reduction formula, and memory-proxy arithmetic.
* **End to end:** generator determinism; all presets contain the required structures; classifier accuracy floor; moving objects tracked with stable IDs; NaN/malformed input; KITTI `.bin` round-trip; REST + binary WebSocket decoding.

Other tools:
```bash
python -m backend.tools.benchmark --frames 100 --preset urban      # CLI benchmark table (+ JSON)
python -m backend.tools.export_sample --preset mixed_traffic       # writes a KITTI .bin sequence to data/sample/
```

---

## 9. Five-minute judge demo

1. **(0:00) Open the dashboard.** The header shows *STREAMING*, live FPS, latency and frame number. The amber badge states that segmentation is the rule-based fallback.
2. **(0:30) Point cloud (3D) tab.** A 64-beam scan with ring patterns and occlusion shadows. Switch colour between *class* and *height*, and try *top-down*.
3. **(1:00) Adaptive 2.5D map tab.** The sensor is at the centre, forward is up. Walk through the legend: drivable, non-drivable, static obstacles, vehicles, pedestrians.
4. **(1:30) Click "10 m".** Inside the cyan ring the cells are 5 cm and the individual squares are visible. Turn on **Grid debug** to see the 0.5 m alignment-tile staircase where the level changes.
5. **(2:00) Click "30 m" → "100 m".** Cells step up to 10, 25 and 50 cm. Set **Colour by → Resolution level** to show the four bands.
6. **(2:30) Hover a cell, then click to pin it.** The inspector shows level, size, key, distance, point count, min/max/mean z, roughness, confidence, dynamic score and the class histogram. Pick a far 50 cm cell holding a car to show min/max z and the mixed histogram.
7. **(3:00) Moving traffic.** White arrows are velocity vectors and dotted lines are trails. The object list's *Moving* filter shows track IDs and speeds. Optionally switch the preset to *Mixed dynamic traffic*.
8. **(3:30) Adaptive vs uniform tab.** The same frame and labels are drawn side by side. Point to the KPI strip: 12.57 M dense 5 cm cells versus about 40 k adaptive, a reduction of about 99.7 %, and the memory proxy in MB.
9. **(4:00) Bottom charts and "Benchmark report".** Latency per frame, cells per frame, cells per band and memory. Run the 100-frame benchmark and show the median/p95 table and the latency-vs-points sweep.
10. **(4:30) 2.5D relief tab, then Architecture tab.** Columns show elevation and overhangs, such as a pedestrian standing under a tree canopy. Finish on the pipeline, live stage timings and the plug-in snippet: *a trained PointNet++ or sparse-conv model replaces the fallback without touching the mapping engine.*

Extras if asked: change the bands live (e.g. the *Aggressive* preset, or enter 0.37/0.41 to show the validation error). Upload a `.bin` file. Scrub back through frame history while paused.

---

## 10. Scope and limitations (stated plainly)

* The segmentation shown is a **rule-based fallback**, not a trained network. The neural adapter is implemented but ships without weights.
* The tracker is a transparent prototype: greedy gated nearest-neighbour association plus a net-displacement motion test. It is **not** a validated production MOT (no Kalman covariance, no Hungarian assignment, no re-identification).
* The map is built per scan in the sensor-centred frame. Multi-frame accumulation with ego-motion compensation is future work.
* Synthetic terrain uses exact object occlusion but not terrain self-occlusion.
* The memory proxy is an estimate (packed record size × cells). Process RSS is reported separately.

## 11. Future work
Trained PointNet++ / KPConv / sparse-conv model; GPU aggregation (PyTorch / CUDA scatter); uncertainty-driven resolution (refine where the classifier is unsure, not only by distance); Bayesian occupancy and semantic fusion over time with ego-motion; Kalman / Hungarian tracking with motion prediction; ROS 2 message interface and edge deployment; camera / radar fusion; robustness to rain, fog and sparse returns.
