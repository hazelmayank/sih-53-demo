import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { X } from "lucide-react";
import { api, type BenchmarkResponse, type Stat } from "../lib/api";
import { fmtCompact, fmtInt } from "../lib/grid";
import { Button } from "./ui";

const TT = {
  contentStyle: { background: "#0c1118", border: "1px solid #26344a", borderRadius: 6, fontSize: 11 },
  labelStyle: { color: "#9aa9bd" },
};
const AXIS = { stroke: "#475569", tick: { fill: "#64748b" }, tickLine: false, fontSize: 10 } as const;

function StatRow({ name, a, b, unit, digits = 1 }: { name: string; a: Stat | number; b?: Stat | number; unit: string; digits?: number }) {
  const f = (s: Stat | number | undefined, k: "median" | "p95") =>
    s === undefined || (typeof s === "number" && !Number.isFinite(s)) ? "–" : typeof s === "number" ? s.toLocaleString(undefined, { maximumFractionDigits: digits }) : s[k].toLocaleString(undefined, { maximumFractionDigits: digits });
  return (
    <tr className="border-t border-line">
      <td className="py-1 pr-3 text-ink-2">{name}</td>
      <td className="num py-1 pr-3 text-right">{f(a, "median")}</td>
      <td className="num py-1 pr-3 text-right text-ink-3">{f(a, "p95")}</td>
      <td className="num py-1 pr-3 text-right">{f(b, "median")}</td>
      <td className="num py-1 text-right text-ink-3">{f(b, "p95")}</td>
      <td className="py-1 pl-2 text-ink-3">{unit}</td>
    </tr>
  );
}

export default function BenchmarkModal({ onClose }: { onClose: () => void }) {
  const [state, setState] = useState<BenchmarkResponse | null>(null);
  const [frames, setFrames] = useState(100);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const s = await api.benchmark();
        if (alive) setState(s);
      } catch {
        /* backend unavailable */
      }
    };
    poll();
    const t = window.setInterval(poll, 600);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, []);

  const run = state?.run;
  const r = run?.result;
  const s = r?.summary;

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-6" onClick={onClose}>
      <div className="scroll-thin max-h-full w-[1100px] max-w-full overflow-y-auto rounded-xl border border-line-2 bg-panel p-5" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-start justify-between">
          <div>
            <h2 className="text-[16px] font-semibold">Adaptive vs uniform benchmark</h2>
            <p className="mt-0.5 max-w-3xl text-[12px] text-ink-3">
              Runs the full pipeline offline on identical synthetic frames with the current configuration. Both maps are built from the
              same points and labels. Medians are reported, and a density sweep shows scaling with point count. Every number here is
              measured on this machine.
            </p>
          </div>
          <button onClick={onClose} className="text-ink-3 hover:text-ink">
            <X size={18} />
          </button>
        </div>

        <div className="mb-4 flex items-center gap-3">
          <label className="flex items-center gap-2 text-[12px] text-ink-2">
            Frames
            <input type="number" min={5} max={500} value={frames} onChange={(e) => setFrames(Number(e.target.value))} className="w-20" />
          </label>
          <Button
            variant="primary"
            disabled={run?.running}
            onClick={async () => {
              setErr(null);
              try {
                await api.runBenchmark(frames, true);
              } catch (e) {
                setErr(String(e));
              }
            }}
          >
            {run?.running ? "Running…" : "Run benchmark"}
          </Button>
          {run?.running && (
            <div className="flex flex-1 items-center gap-2">
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-line">
                <div className="h-full bg-accent transition-all" style={{ width: `${(run.progress * 100).toFixed(0)}%` }} />
              </div>
              <span className="num text-[11px] text-ink-3">{run.message}</span>
            </div>
          )}
          {(err || run?.error) && <span className="text-[12px] text-bad">{err ?? run?.error}</span>}
        </div>

        {!r && !run?.running && <div className="rounded-md border border-dashed border-line-2 p-6 text-center text-[12.5px] text-ink-3">No benchmark run yet.</div>}

        {r && s && (
          <>
            <div className="mb-4 grid grid-cols-4 gap-2">
              {[
                ["Cell reduction vs dense 5 cm", `${s.reduction_vs_dense_pct.median.toFixed(2)}%`, "#34d399"],
                ["Cell reduction vs sparse 5 cm", `${s.reduction_vs_sparse_pct.median.toFixed(1)}%`, "#34d399"],
                ["Median end-to-end latency", `${s.latency_ms.median.toFixed(1)} ms`, "#a78bfa"],
                ["Alignment check (all frames)", s.partition_ok_all ? "PASS" : "FAIL", s.partition_ok_all ? "#34d399" : "#f87171"],
              ].map(([k, v, c]) => (
                <div key={k} className="rounded-lg border border-line bg-bg px-3 py-2">
                  <div className="text-[9.5px] uppercase tracking-[0.12em] text-ink-3">{k}</div>
                  <div className="num text-[20px] font-semibold" style={{ color: c }}>
                    {v}
                  </div>
                </div>
              ))}
            </div>

            <table className="mb-4 w-full text-[12px]">
              <thead>
                <tr className="text-[10px] uppercase tracking-wider text-ink-3">
                  <th className="pb-1 text-left font-medium">metric ({r.frames} frames)</th>
                  <th className="pb-1 pr-3 text-right font-medium text-accent">adaptive median</th>
                  <th className="pb-1 pr-3 text-right font-medium">p95</th>
                  <th className="pb-1 pr-3 text-right font-medium text-orange-400">uniform 5 cm median</th>
                  <th className="pb-1 text-right font-medium">p95</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                <StatRow name="Allocated cells (sparse)" a={s.adaptive_cells} b={s.uniform_sparse_cells} unit="cells" digits={0} />
                <StatRow name="Cells if allocated densely (100 m disk)" a={NaN} b={s.uniform_dense_cells} unit="cells" digits={0} />
                <StatRow name="Memory proxy (sparse)" a={s.adaptive_mb} b={s.uniform_sparse_mb} unit="MB" digits={2} />
                <StatRow name="Memory proxy (dense grid)" a={NaN} b={s.uniform_dense_mb} unit="MB" digits={0} />
                <StatRow name="Projection + aggregation time" a={s.adaptive_projection_ms} b={s.uniform_projection_ms} unit="ms" />
                <StatRow name="Points per frame" a={s.points} unit="points" digits={0} />
                {s.seg_accuracy && <StatRow name="Fallback segmentation accuracy vs GT" a={{ ...s.seg_accuracy, median: s.seg_accuracy.median * 100, p95: s.seg_accuracy.p95 * 100 }} unit="%" />}
              </tbody>
            </table>

            <div className="grid grid-cols-2 gap-3">
              <div className="h-60 rounded-lg border border-line bg-bg p-2">
                <div className="panel-title mb-1 px-1">Latency vs point count (density sweep)</div>
                <ResponsiveContainer width="100%" height="88%">
                  <LineChart data={r.scaling} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
                    <CartesianGrid stroke="#1c2633" />
                    <XAxis dataKey="points" {...AXIS} tickFormatter={fmtCompact} />
                    <YAxis {...AXIS} />
                    <Tooltip {...TT} labelFormatter={(v) => `${fmtInt(Number(v))} points`} formatter={(v: number) => v.toFixed(1) + " ms"} />
                    <Legend iconSize={8} wrapperStyle={{ fontSize: 10.5 }} />
                    <Line dataKey="latency_ms" name="end-to-end" stroke="#a78bfa" />
                    <Line dataKey="adaptive_projection_ms" name="adaptive projection" stroke="#22d3ee" />
                    <Line dataKey="uniform_projection_ms" name="uniform projection" stroke="#f97316" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="h-60 rounded-lg border border-line bg-bg p-2">
                <div className="panel-title mb-1 px-1">Allocated cells vs point count</div>
                <ResponsiveContainer width="100%" height="88%">
                  <LineChart data={r.scaling} margin={{ top: 4, right: 12, left: -4, bottom: 0 }}>
                    <CartesianGrid stroke="#1c2633" />
                    <XAxis dataKey="points" {...AXIS} tickFormatter={fmtCompact} />
                    <YAxis {...AXIS} tickFormatter={fmtCompact} />
                    <Tooltip {...TT} labelFormatter={(v) => `${fmtInt(Number(v))} points`} formatter={(v: number) => fmtInt(v)} />
                    <Legend iconSize={8} wrapperStyle={{ fontSize: 10.5 }} />
                    <Line dataKey="adaptive_cells" name="adaptive" stroke="#22d3ee" />
                    <Line dataKey="uniform_sparse_cells" name="uniform 5 cm (sparse)" stroke="#f97316" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="mt-3 text-[11px] text-ink-3">
              Run {r.created} · {r.duration_s.toFixed(1)} s · preset {String(r.config.preset)} · {String(r.config.beams)} beams
              {r.saved_to && (
                <>
                  {" "}· saved to <span className="num text-ink-2">{r.saved_to}</span>
                </>
              )}
              . Memory proxy = packed 58-byte cell record (+16 B sparse index), not measured DRAM; process RSS is shown separately.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
