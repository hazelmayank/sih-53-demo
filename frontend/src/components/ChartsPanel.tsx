import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SeriesPoint } from "../hooks/useLidarStream";
import type { Frame } from "../types";
import { LEVEL_COLORS } from "../lib/colors";
import { fmtCompact } from "../lib/grid";

const ADAPT = "#22d3ee";
const UNI = "#f97316";
const DENSE = "#ef4444";
const GRID = "#1c2633";
const AXIS = { stroke: "#475569", tick: { fill: "#64748b" }, tickLine: false } as const;
const TT = {
  contentStyle: { background: "#0c1118", border: "1px solid #26344a", borderRadius: 6, fontSize: 11 },
  labelStyle: { color: "#9aa9bd" },
  itemStyle: { padding: 0 },
};

function ChartCard({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <div className="flex min-w-0 flex-1 flex-col rounded-lg border border-line bg-panel px-2.5 pb-1 pt-2">
      <div className="flex items-baseline justify-between px-1">
        <span className="panel-title">{title}</span>
        {note && <span className="text-[10px] text-ink-3">{note}</span>}
      </div>
      <div className="min-h-0 flex-1">{children}</div>
    </div>
  );
}

export default function ChartsPanel({ series, frame }: { series: SeriesPoint[]; frame: Frame | null }) {
  const cmp = frame?.header.comparison;
  const profile =
    cmp?.band_profile.map((b) => ({
      name: b.label,
      level: b.level,
      adaptive: b.adaptive_cells,
      uniform: b.uniform_sparse_cells,
      size: b.cell_size_m,
    })) ?? [];
  const mem = cmp
    ? [
        { name: "Adaptive", mb: cmp.adaptive_memory_proxy_mb, color: ADAPT },
        { name: "Uniform sparse", mb: cmp.uniform_sparse_memory_proxy_mb, color: UNI },
        { name: "Uniform dense", mb: cmp.uniform_dense_memory_proxy_mb, color: DENSE },
      ]
    : [];

  return (
    <div className="flex h-full gap-2">
      <ChartCard title="Processing time / frame" note="measured, ms">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={series} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="frame" {...AXIS} fontSize={10} minTickGap={30} />
            <YAxis {...AXIS} fontSize={10} />
            <Tooltip {...TT} formatter={(v: number) => v.toFixed(1) + " ms"} />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 10.5 }} />
            <Line isAnimationActive={false} dot={false} dataKey="latency" name="end-to-end latency" stroke="#a78bfa" strokeWidth={1.5} />
            <Line isAnimationActive={false} dot={false} dataKey="adaptiveMs" name="adaptive projection" stroke={ADAPT} strokeWidth={1.5} />
            <Line isAnimationActive={false} dot={false} dataKey="uniformMs" name="uniform 5 cm projection" stroke={UNI} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Allocated cells / frame" note="same points, same labels">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={series} margin={{ top: 6, right: 8, left: -8, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="frame" {...AXIS} fontSize={10} minTickGap={30} />
            <YAxis {...AXIS} fontSize={10} tickFormatter={fmtCompact} />
            <Tooltip {...TT} formatter={(v: number) => v.toLocaleString()} />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 10.5 }} />
            <Area isAnimationActive={false} dataKey="uniformCells" name="uniform 5 cm (sparse)" stroke={UNI} fill={UNI} fillOpacity={0.12} />
            <Area isAnimationActive={false} dataKey="adaptiveCells" name="adaptive" stroke={ADAPT} fill={ADAPT} fillOpacity={0.2} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Range profile · cells per band" note="log scale">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={profile} margin={{ top: 6, right: 8, left: -8, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="name" {...AXIS} fontSize={10} />
            <YAxis {...AXIS} fontSize={10} scale="log" domain={[1, "auto"]} allowDataOverflow tickFormatter={fmtCompact} />
            <Tooltip {...TT} formatter={(v: number) => v.toLocaleString()} />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 10.5 }} />
            <Bar isAnimationActive={false} dataKey="uniform" name="uniform 5 cm" fill={UNI} fillOpacity={0.8} />
            <Bar isAnimationActive={false} dataKey="adaptive" name="adaptive">
              {profile.map((p) => (
                <Cell key={p.level} fill={LEVEL_COLORS[p.level % LEVEL_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Memory proxy · current frame" note="packed cell records, log MB">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={mem} layout="vertical" margin={{ top: 6, right: 40, left: 20, bottom: 0 }}>
            <CartesianGrid stroke={GRID} horizontal={false} />
            <XAxis type="number" {...AXIS} fontSize={10} scale="log" domain={[0.1, "auto"]} allowDataOverflow />
            <YAxis type="category" dataKey="name" {...AXIS} fontSize={10.5} width={82} />
            <Tooltip {...TT} formatter={(v: number) => v.toFixed(2) + " MB"} />
            <Bar isAnimationActive={false} dataKey="mb" label={{ position: "right", fill: "#9aa9bd", fontSize: 10.5, formatter: (v: number) => (v >= 10 ? v.toFixed(0) : v.toFixed(2)) + " MB" }}>
              {mem.map((m) => (
                <Cell key={m.name} fill={m.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}
