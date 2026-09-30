import type { Frame } from "../types";
import { fmtInt, fmtSize } from "../lib/grid";
import { LEVEL_COLORS } from "../lib/colors";

const STAGES: { key: string; title: string; module: string; text: string }[] = [
  { key: "input", title: "LiDAR input", module: "lidar/synthetic_scene.py · lidar/io.py", text: "Ray-cast 64-beam spinning sensor (deterministic seeds) or imported .bin / .pcd / .ply sequences. Same internal point format either way." },
  { key: "preprocess", title: "Preprocessing", module: "lidar/preprocess.py", text: "Sanitise malformed data, range gate [min, max], isolated-return outlier filter (27-voxel neighbourhood), optional voxel downsampling." },
  { key: "ground", title: "Terrain estimation", module: "lidar/preprocess.py", text: "Polar line-tracking ground model: per-sector march with slope + vertical-structure gating. Gives height above ground and slope." },
  { key: "clustering", title: "2.5D clustering", module: "lidar/clustering.py", text: "Body / overhead layers labelled separately, joined only where vertically continuous, so a pedestrian under a tree stays separate." },
  { key: "segmentation", title: "Semantic inference", module: "lidar/segmentation.py", text: "Pluggable SegmentationModel. Here: a rule-based fallback. A TorchScript PointNet++ / SparseConv adapter is ready." },
  { key: "tracking", title: "Objects + tracking", module: "tracking/tracker.py", text: "Oriented boxes, gated nearest-neighbour association, smoothed velocity and a net-displacement motion test." },
  { key: "adaptive_map", title: "Adaptive 2.5D map", module: "mapping/adaptive_grid.py · projection.py", text: "Range → level per 0.5 m alignment tile, CellKey(level, ix, iy), sparse hash aggregation of z stats + class histograms." },
  { key: "uniform_baseline", title: "Uniform baseline", module: "mapping/baseline.py", text: "Same points and labels into a 5 cm grid, plus an exact dense cell count, for a like-for-like comparison." },
  { key: "metrics", title: "Metrics + stream", module: "core/metrics.py · api/protocol.py", text: "Latency, FPS, cells, memory proxy and alignment check. Binary typed-array WebSocket frames go to this dashboard." },
];

export default function ArchitectureView({ frame }: { frame: Frame | null }) {
  const st = frame?.header.stage_ms ?? {};
  const total = Object.values(st).reduce((a, b) => a + b, 0) || 1;
  const sched = frame?.header.schedule;
  return (
    <div className="scroll-thin h-full overflow-y-auto p-5">
      <div className="mb-4 flex items-end justify-between">
        <div>
          <h2 className="text-[16px] font-semibold">Pipeline architecture</h2>
          <p className="text-[12px] text-ink-3">Every stage is a separate, tested module. Timings below are measured live on the current frame.</p>
        </div>
        {frame && (
          <div className="num text-right text-[12px] text-ink-2">
            frame #{frame.header.frame_id} · {fmtInt(frame.header.metrics.points)} points · {frame.header.metrics.latency_ms.toFixed(1)} ms to map-ready
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3">
        {STAGES.map((s, i) => {
          const ms = s.key === "input" ? frame?.header.metrics.sensor_sim_ms : st[s.key];
          const hot = s.key === "adaptive_map";
          return (
            <div key={s.key} className={`relative rounded-lg border p-3 ${hot ? "border-accent/60 bg-accent/[0.06]" : "border-line bg-panel"}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="num grid h-5 w-5 place-items-center rounded bg-panel-2 text-[10.5px] text-ink-3">{i + 1}</span>
                  <span className={`text-[13px] font-semibold ${hot ? "text-accent" : ""}`}>{s.title}</span>
                </div>
                <span className="num text-[12px] text-ink-2">{ms !== undefined ? `${ms.toFixed(1)} ms` : "–"}</span>
              </div>
              <div className="num mt-0.5 text-[10.5px] text-ink-3">{s.module}</div>
              <p className="mt-1.5 text-[12px] leading-snug text-ink-2">{s.text}</p>
              {s.key !== "input" && ms !== undefined && (
                <div className="mt-2 h-1 overflow-hidden rounded-full bg-line">
                  <div className="h-full rounded-full bg-accent/70" style={{ width: `${Math.min(100, (100 * ms) / total)}%` }} />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {sched && (
        <div className="mt-5 grid grid-cols-[1.2fr_1fr] gap-4">
          <div className="rounded-lg border border-line bg-panel p-4">
            <div className="panel-title mb-2">Active resolution schedule</div>
            <table className="w-full text-[12px]">
              <thead className="text-[10px] uppercase tracking-wider text-ink-3">
                <tr>
                  <th className="text-left font-medium">level</th>
                  <th className="text-left font-medium">range band</th>
                  <th className="text-right font-medium">cell</th>
                  <th className="text-right font-medium">cells / tile</th>
                  <th className="text-right font-medium">allocated</th>
                  <th className="text-right font-medium">dense would be</th>
                </tr>
              </thead>
              <tbody>
                {sched.levels.map((l) => {
                  const bp = frame!.header.comparison.band_profile[l.level];
                  return (
                    <tr key={l.level} className="border-t border-line">
                      <td className="py-1" style={{ color: LEVEL_COLORS[l.level % LEVEL_COLORS.length] }}>
                        L{l.level}
                      </td>
                      <td className="num">[{l.lower_m}, {l.upper_m}) m</td>
                      <td className="num text-right">{fmtSize(l.cell_size_m)}</td>
                      <td className="num text-right">{l.subdiv}×{l.subdiv}</td>
                      <td className="num text-right">{fmtInt(bp?.adaptive_cells)}</td>
                      <td className="num text-right text-ink-3">{fmtInt(bp?.adaptive_dense_cells)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-2 text-[11.5px] leading-snug text-ink-3">
              Alignment tile = {fmtSize(sched.tile_m)} (LCM of the cell sizes). Quantum = {fmtSize(sched.quantum_m)} (their GCD). Each tile gets exactly
              one level from its centre's range, and every level's grid is anchored at the sensor origin. So the cells of any level tile it
              exactly, and the map is a gap-free, overlap-free partition. That is checked on every frame.
            </p>
          </div>
          <div className="rounded-lg border border-line bg-panel p-4">
            <div className="panel-title mb-2">Plug in a trained network</div>
            <pre className="num overflow-x-auto rounded bg-bg p-3 text-[11px] leading-relaxed text-ink-2">{`class MyPointNet(SegmentationModel):
    name, kind = "pointnet2", "neural"
    def predict(self, frame, ctx):
        logits = self.net(frame.xyz, frame.intensity)
        return SegmentationResult(labels, confidence)

# or, no code: export TorchScript and set
#   processing.segmenter  = "torchscript"
#   processing.model_path = "models/pn2.pt"`}</pre>
            <p className="mt-2 text-[11.5px] leading-snug text-ink-3">
              The adaptive grid, baseline, tracker and dashboard consume only per-point labels and confidences. Nothing downstream changes.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
