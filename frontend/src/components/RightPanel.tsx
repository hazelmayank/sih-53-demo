import { useEffect, useState } from "react";
import { MousePointer2 } from "lucide-react";
import { api } from "../lib/api";
import { CLASS_COLORS, CLASS_NAMES, CLASS_SHORT, LEVEL_COLORS } from "../lib/colors";
import { fmtInt, fmtSize } from "../lib/grid";
import type { CellDetail, Frame, PickedCell } from "../types";
import { Dot, Row, Section, Segmented } from "./ui";

function Bar({ value, max, color }: { value: number; max: number; color: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-line">
      <div className="h-full rounded-full" style={{ width: `${max > 0 ? (100 * value) / max : 0}%`, background: color }} />
    </div>
  );
}

function CellInspector({ cell, pinned }: { cell: PickedCell | null; pinned: boolean }) {
  const [detail, setDetail] = useState<CellDetail | null>(null);
  useEffect(() => {
    setDetail(null);
    if (!cell || cell.mode !== "adaptive") return;
    const t = window.setTimeout(() => {
      api.cell(cell.frameId, cell.level, cell.ix, cell.iy).then(setDetail).catch(() => setDetail(null));
    }, 90);
    return () => window.clearTimeout(t);
  }, [cell?.frameId, cell?.level, cell?.ix, cell?.iy, cell?.mode]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!cell)
    return (
      <div className="flex items-center gap-2 rounded-md border border-dashed border-line-2 px-3 py-4 text-[12px] text-ink-3">
        <MousePointer2 size={15} /> Hover a cell on the map, click to pin it.
      </div>
    );
  const hist = detail ? Object.entries(detail.class_histogram).map(([k, v]) => [Number(k), v] as const) : [];
  const hmax = Math.max(1, ...hist.map(([, v]) => v));
  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <span className="rounded px-1.5 py-0.5 text-[11px] font-semibold" style={{ background: (cell.mode === "adaptive" ? LEVEL_COLORS[cell.level % LEVEL_COLORS.length] : "#94a3b8") + "22", color: cell.mode === "adaptive" ? LEVEL_COLORS[cell.level % LEVEL_COLORS.length] : "#94a3b8" }}>
          {cell.mode === "adaptive" ? `LEVEL ${cell.level}` : "UNIFORM"} · {fmtSize(cell.cellSize)}
        </span>
        <span className="flex items-center gap-1.5 text-[12px]">
          <Dot color={CLASS_COLORS[cell.label]} /> {CLASS_NAMES[cell.label]}
        </span>
        {pinned && <span className="ml-auto text-[10px] uppercase tracking-wider text-accent">pinned</span>}
      </div>
      <Row k="Key (level, ix, iy)" v={`(${cell.level}, ${cell.ix}, ${cell.iy})`} />
      <Row k="Centre (x, y)" v={`${cell.cx.toFixed(2)}, ${cell.cy.toFixed(2)} m`} />
      <Row k="Distance" v={`${cell.range.toFixed(2)} m`} />
      {cell.count !== undefined && <Row k="Point count" v={fmtInt(cell.count)} />}
      {cell.zmin !== undefined && <Row k="min / mean / max z" v={`${cell.zmin.toFixed(2)} / ${cell.zmean!.toFixed(2)} / ${cell.zmax.toFixed(2)} m`} />}
      {cell.zmin === undefined && <Row k="max z" v={`${cell.zmax.toFixed(2)} m`} />}
      {cell.rough !== undefined && <Row k="Roughness (σz)" v={`${(cell.rough * 100).toFixed(1)} cm`} />}
      {cell.conf !== undefined && <Row k="Confidence" v={`${(cell.conf * 100).toFixed(0)}%`} />}
      {cell.dyn !== undefined && <Row k="Dynamic score" v={cell.dyn.toFixed(2)} color={cell.dyn > 0.3 ? "#ff4f8b" : undefined} />}
      {cell.occ !== undefined && <Row k="Occupancy" v={cell.occ.toFixed(2)} />}
      {detail && <Row k="Last seen frame" v={`#${detail.last_seen_frame}`} />}
      {cell.mode === "adaptive" && (
        <div className="mt-2">
          <div className="mb-1 text-[10.5px] uppercase tracking-wider text-ink-3">Class histogram</div>
          {!detail && <div className="text-[11px] text-ink-3">loading…</div>}
          {hist.map(([k, v]) => (
            <div key={k} className="mb-1 grid grid-cols-[34px_1fr_34px] items-center gap-2 text-[11px]">
              <span style={{ color: CLASS_COLORS[k] }}>{CLASS_SHORT[k]}</span>
              <Bar value={v} max={hmax} color={CLASS_COLORS[k]} />
              <span className="num text-right text-ink-2">{v}</span>
            </div>
          ))}
        </div>
      )}
      {cell.mode === "uniform" && <div className="mt-2 text-[11px] text-ink-3">Uniform baseline cells stream only class and max z.</div>}
    </div>
  );
}

function SegmentationStats({ frame }: { frame: Frame }) {
  const ev = frame.header.segmentation_eval;
  const stats = frame.header.class_stats;
  const pmax = Math.max(1, ...stats.map((s) => s.points));
  return (
    <div>
      <div className="mb-1.5 grid grid-cols-[1fr_56px_52px_40px] gap-2 text-[10px] uppercase tracking-wider text-ink-3">
        <span>class</span>
        <span className="text-right">points</span>
        <span className="text-right">cells</span>
        <span className="text-right">IoU</span>
      </div>
      {stats.map((s) => (
        <div key={s.id} className="mb-1">
          <div className="grid grid-cols-[1fr_56px_52px_40px] items-center gap-2 text-[11.5px]">
            <span className="flex items-center gap-1.5 truncate">
              <Dot color={CLASS_COLORS[s.id]} /> {CLASS_NAMES[s.id]}
            </span>
            <span className="num text-right text-ink-2">{fmtInt(s.points)}</span>
            <span className="num text-right text-ink-2">{fmtInt(s.cells)}</span>
            <span className="num text-right text-ink-2">{ev?.iou[s.id] != null ? ev.iou[s.id]!.toFixed(2) : "–"}</span>
          </div>
          <Bar value={s.points} max={pmax} color={CLASS_COLORS[s.id]} />
        </div>
      ))}
      <div className="mt-2 grid grid-cols-3 gap-1.5">
        <div className="rounded border border-line bg-bg px-2 py-1">
          <div className="text-[9.5px] uppercase tracking-wider text-ink-3">Mean conf.</div>
          <div className="num text-[13px]">{(frame.header.metrics.class_confidence * 100).toFixed(0)}%</div>
        </div>
        <div className="rounded border border-line bg-bg px-2 py-1" title="Point-wise accuracy against the synthetic ground truth">
          <div className="text-[9.5px] uppercase tracking-wider text-ink-3">Acc. vs GT</div>
          <div className="num text-[13px]">{ev ? `${(ev.accuracy * 100).toFixed(1)}%` : "n/a"}</div>
        </div>
        <div className="rounded border border-line bg-bg px-2 py-1">
          <div className="text-[9.5px] uppercase tracking-wider text-ink-3">mIoU vs GT</div>
          <div className="num text-[13px]">{ev ? ev.miou.toFixed(2) : "n/a"}</div>
        </div>
      </div>
      {!ev && <div className="mt-1 text-[10.5px] text-ink-3">Ground truth only exists for synthetic scenes (and is never used by the fallback classifier).</div>}
    </div>
  );
}

function ObjectList({ frame, onFocus }: { frame: Frame; onFocus: (x: number, y: number) => void }) {
  const [filter, setFilter] = useState<"moving" | "agents" | "all">("agents");
  const objs = frame.header.objects.filter((o) =>
    filter === "moving" ? o.dynamic : filter === "agents" ? o.class === 3 || o.class === 4 : true,
  );
  return (
    <div>
      <Segmented
        size="xs"
        value={filter}
        options={[
          { value: "agents", label: "Vehicles + peds" },
          { value: "moving", label: "Moving" },
          { value: "all", label: "All" },
        ]}
        onChange={setFilter}
      />
      <div className="mt-2 grid grid-cols-[42px_1fr_48px_54px] gap-2 px-1 text-[10px] uppercase tracking-wider text-ink-3">
        <span>track</span>
        <span>class</span>
        <span className="text-right">dist</span>
        <span className="text-right">motion</span>
      </div>
      <div className="mt-1 space-y-0.5">
        {objs.slice(0, 60).map((o) => (
          <button
            key={o.id}
            onClick={() => onFocus(o.x, o.y)}
            className="grid w-full grid-cols-[42px_1fr_48px_54px] items-center gap-2 rounded px-1 py-[3px] text-left text-[11.5px] hover:bg-panel-2"
          >
            <span className="num text-ink-2">{o.track_id != null ? `#${o.track_id}` : "–"}</span>
            <span className="flex items-center gap-1.5 truncate">
              <Dot color={CLASS_COLORS[o.class]} size={8} />
              <span className="truncate">{CLASS_NAMES[o.class]}</span>
              <span className="num text-[10px] text-ink-3">{(o.confidence * 100).toFixed(0)}%</span>
            </span>
            <span className="num text-right text-ink-2">{o.distance.toFixed(1)}m</span>
            <span className={`num text-right ${o.dynamic ? "text-[#ff4f8b]" : "text-ink-3"}`}>{o.dynamic ? `${o.speed.toFixed(1)} m/s` : "static"}</span>
          </button>
        ))}
        {objs.length === 0 && <div className="px-1 py-2 text-[11px] text-ink-3">No objects for this filter.</div>}
        {objs.length > 60 && <div className="px-1 text-[10.5px] text-ink-3">+{objs.length - 60} more</div>}
      </div>
    </div>
  );
}

export default function RightPanel({ frame, cell, pinned, onFocus }: { frame: Frame | null; cell: PickedCell | null; pinned: boolean; onFocus: (x: number, y: number) => void }) {
  return (
    <aside className="scroll-thin flex w-[320px] shrink-0 flex-col overflow-y-auto border-l border-line bg-panel">
      <Section title="Cell inspector">
        <CellInspector cell={cell} pinned={pinned} />
      </Section>
      {frame && (
        <Section title="Semantics" right={<span className="text-[10.5px] text-ink-3">{frame.header.segmenter.kind === "fallback" ? "rule-based" : frame.header.segmenter.name}</span>}>
          <SegmentationStats frame={frame} />
        </Section>
      )}
      {frame && (
        <Section title={`Objects (${frame.header.objects.length})`} right={<span className="text-[10.5px] text-ink-3">NN tracker · gated</span>}>
          <ObjectList frame={frame} onFocus={onFocus} />
        </Section>
      )}
    </aside>
  );
}
