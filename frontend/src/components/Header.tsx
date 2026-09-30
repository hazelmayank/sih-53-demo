import { Activity, Cpu, Gauge, Radar, Timer } from "lucide-react";
import type { Frame } from "../types";
import { fmtInt } from "../lib/grid";

function Metric({ icon, label, value, unit, tone }: { icon: React.ReactNode; label: string; value: string; unit?: string; tone?: string }) {
  return (
    <div className="flex items-center gap-2 border-l border-line px-3.5">
      <span className="text-ink-3">{icon}</span>
      <div className="leading-tight">
        <div className="text-[9.5px] uppercase tracking-[0.12em] text-ink-3">{label}</div>
        <div className="num text-[14px] font-semibold" style={{ color: tone ?? "var(--color-ink)" }}>
          {value}
          {unit && <span className="ml-0.5 text-[10.5px] font-normal text-ink-3">{unit}</span>}
        </div>
      </div>
    </div>
  );
}

export default function Header({ frame, connected }: { frame: Frame | null; connected: boolean }) {
  const st = frame?.header.status;
  const m = frame?.header.metrics;
  const state = !connected ? "OFFLINE" : st?.error ? "ERROR" : st?.running ? "STREAMING" : "PAUSED";
  const tone = { OFFLINE: "#f87171", ERROR: "#f87171", STREAMING: "#34d399", PAUSED: "#fbbf24" }[state];
  const seg = frame?.header.segmenter;
  return (
    <header className="flex h-[52px] shrink-0 items-center border-b border-line bg-panel">
      <div className="flex items-center gap-3 px-4">
        <div className="grid h-8 w-8 place-items-center rounded-lg border border-accent/40 bg-accent/10">
          <Radar size={17} className="text-accent" />
        </div>
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight">Adaptive 2.5D LiDAR Mapping</div>
          <div className="text-[10.5px] text-ink-3">SIH PS 26053 · DRDO · foveated variable-resolution perception</div>
        </div>
      </div>
      <div className="ml-2 flex items-center gap-2 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold tracking-wider" style={{ borderColor: tone + "66", color: tone, background: tone + "12" }}>
        <span className={`h-1.5 w-1.5 rounded-full ${state === "STREAMING" ? "animate-pulse" : ""}`} style={{ background: tone }} />
        {state}
      </div>
      {seg && (
        <div
          className={`ml-2 rounded-full border px-2.5 py-0.5 text-[10.5px] font-medium ${seg.kind === "fallback" ? "border-warn/50 bg-warn/10 text-warn" : seg.kind === "oracle" ? "border-violet-400/50 bg-violet-400/10 text-violet-300" : "border-ok/50 bg-ok/10 text-ok"}`}
          title={seg.description}
        >
          {seg.kind === "fallback" ? "Segmentation: rule-based fallback (not a trained NN)" : seg.kind === "oracle" ? "Segmentation: ground-truth oracle (debug)" : `Segmentation: ${seg.name}`}
        </div>
      )}
      <div className="ml-auto flex h-full items-center">
        <Metric icon={<Gauge size={15} />} label="Output FPS" value={m ? m.fps.toFixed(1) : "–"} tone={m && m.fps > 0 ? "#34d399" : undefined} />
        <Metric icon={<Timer size={15} />} label="Latency" value={m ? m.latency_ms.toFixed(1) : "–"} unit="ms" />
        <Metric icon={<Cpu size={15} />} label="Capacity" value={m ? m.capacity_fps.toFixed(0) : "–"} unit="fps" />
        <Metric icon={<Activity size={15} />} label="Frame" value={frame ? `#${frame.header.frame_id}` : "–"} />
        <Metric icon={<Radar size={15} />} label="Sensor" value={frame ? frame.header.source.replace("synthetic:", "sim · ").replace("file:", "file · ") : "–"} />
        <Metric icon={<Activity size={15} />} label="Points" value={fmtInt(m?.points)} />
      </div>
    </header>
  );
}
