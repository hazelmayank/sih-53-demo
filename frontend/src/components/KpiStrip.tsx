import type { Frame } from "../types";
import { fmtCompact, fmtInt } from "../lib/grid";

function Card({ label, value, sub, tone, big }: { label: string; value: string; sub?: string; tone?: string; big?: boolean }) {
  return (
    <div className={`min-w-0 flex-1 rounded-lg border border-line bg-panel px-3 py-2 ${big ? "border-accent/40 bg-accent/[0.06]" : ""}`}>
      <div className="truncate text-[9.5px] uppercase tracking-[0.12em] text-ink-3">{label}</div>
      <div className="num mt-0.5 truncate text-[19px] font-semibold leading-tight" style={{ color: tone ?? "var(--color-ink)" }}>
        {value}
      </div>
      {sub && <div className="num truncate text-[10.5px] text-ink-3">{sub}</div>}
    </div>
  );
}

export default function KpiStrip({ frame }: { frame: Frame | null }) {
  const m = frame?.header.metrics;
  const c = frame?.header.comparison;
  return (
    <div className="flex gap-2">
      <Card label="Adaptive cells" value={fmtInt(m?.occupied_cells)} sub={m ? `${m.points_per_cell.toFixed(2)} pts / cell` : undefined} tone="#22d3ee" />
      <Card label="Dense 5 cm grid" value={fmtCompact(m?.uniform_dense_cells)} sub="whole 100 m disk" />
      <Card label="Sparse 5 cm grid" value={fmtInt(m?.uniform_sparse_cells)} sub="same points" />
      <Card
        label="Reduction vs dense"
        value={m ? `${m.reduction_vs_dense_pct.toFixed(2)}%` : "–"}
        sub={m ? `vs sparse: ${m.reduction_vs_sparse_pct.toFixed(1)}%` : undefined}
        tone="#34d399"
        big
      />
      <Card
        label="Memory proxy"
        value={m ? `${m.memory_proxy_mb.toFixed(2)} MB` : "–"}
        sub={c ? `dense: ${c.uniform_dense_memory_proxy_mb.toFixed(0)} MB` : undefined}
      />
      <Card
        label="Alignment"
        value={frame ? (frame.header.partition.ok ? "PASS" : "FAIL") : "–"}
        sub={frame ? `${frame.header.partition.mixed_level_tiles + frame.header.partition.misassigned_cells} overlaps · ${fmtCompact(frame.header.partition.tiles)} tiles` : undefined}
        tone={frame ? (frame.header.partition.ok ? "#34d399" : "#f87171") : undefined}
      />
      <Card label="Moving objects" value={m ? String(m.dynamic_objects) : "–"} sub={m ? `${m.objects} obj · ${m.tracks} tracks` : undefined} tone="#ff4f8b" />
    </div>
  );
}
