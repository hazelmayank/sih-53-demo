import { useCallback, useEffect, useMemo, useState } from "react";
import { BarChart3, Boxes, Columns2, Map as MapIcon, Network, ScatterChart } from "lucide-react";
import Header from "../components/Header";
import KpiStrip from "../components/KpiStrip";
import ControlPanel from "../components/ControlPanel";
import RightPanel from "../components/RightPanel";
import MapCanvas, { type ViewState } from "../components/MapCanvas";
import ReliefView from "../components/ReliefView";
import PointCloudView from "../components/PointCloudView";
import ChartsPanel from "../components/ChartsPanel";
import ArchitectureView from "../components/ArchitectureView";
import BenchmarkModal from "../components/BenchmarkModal";
import { useLidarStream } from "../hooks/useLidarStream";
import { api } from "../lib/api";
import { CLASS_COLORS, CLASS_NAMES, LEVEL_COLORS, ramp, rgbCss } from "../lib/colors";
import { fmtInt, fmtSize } from "../lib/grid";
import type { AppConfig, DisplayOptions, PickedCell, ViewTab } from "../types";
import { Dot } from "../components/ui";

const TABS: { id: ViewTab; label: string; icon: React.ReactNode }[] = [
  { id: "map", label: "Adaptive 2.5D map", icon: <MapIcon size={14} /> },
  { id: "split", label: "Adaptive vs uniform", icon: <Columns2 size={14} /> },
  { id: "relief", label: "2.5D relief (3D)", icon: <Boxes size={14} /> },
  { id: "points", label: "Point cloud (3D)", icon: <ScatterChart size={14} /> },
  { id: "architecture", label: "Architecture", icon: <Network size={14} /> },
];

function MapLegend({ display, frame }: { display: DisplayOptions; frame: ReturnType<typeof useLidarStream>["frame"] }) {
  if (display.colorMode === "class")
    return (
      <div className="flex flex-wrap gap-x-3 gap-y-1">
        {CLASS_NAMES.map((n, i) => (
          <span key={n} className={`flex items-center gap-1.5 text-[11px] ${display.classVisible[i] ? "text-ink-2" : "text-ink-3 line-through"}`}>
            <Dot color={CLASS_COLORS[i]} /> {n}
          </span>
        ))}
      </div>
    );
  if (display.colorMode === "level" && frame)
    return (
      <div className="flex flex-wrap gap-x-3">
        {frame.header.schedule.levels.map((l) => (
          <span key={l.level} className="flex items-center gap-1.5 text-[11px] text-ink-2">
            <Dot color={LEVEL_COLORS[l.level % LEVEL_COLORS.length]} /> L{l.level} {fmtSize(l.cell_size_m)}
          </span>
        ))}
      </div>
    );
  const labels = { elevation: ["−1 m", "max z", "5 m"], confidence: ["0%", "confidence", "100%"], dynamic: ["static", "dynamic score", "moving"] }[
    display.colorMode as "elevation" | "confidence" | "dynamic"
  ];
  return (
    <div className="flex items-center gap-2 text-[11px] text-ink-3">
      {labels[0]}
      <div className="h-2 w-40 rounded" style={{ background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => rgbCss(ramp(t))).join(",")})` }} />
      {labels[2]} <span className="text-ink-2">{labels[1]}</span>
    </div>
  );
}

export default function Dashboard() {
  const [tab, setTab] = useState<ViewTab>(() => {
    const t = new URLSearchParams(location.search).get("tab");
    return (TABS.some((x) => x.id === t) ? t : "map") as ViewTab;
  });
  const [display, setDisplay] = useState<DisplayOptions>({
    colorMode: "class",
    showBands: true,
    showGridDebug: false,
    showPoints: false,
    showObjects: true,
    showTracks: true,
    baseline: "adaptive",
    classVisible: [true, true, true, true, true, true],
  });
  const needUniform = tab === "split" || (tab === "map" && display.baseline === "uniform");
  const streamOpts = useMemo(
    () => ({ points: tab === "points" || display.showPoints, uniform: needUniform, max_points: 80000 }),
    [tab, display.showPoints, needUniform],
  );
  const { frame, connected, series, requestFrame } = useLidarStream(streamOpts);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [cfgError, setCfgError] = useState<string | null>(null);
  const [view, setView] = useState<ViewState>({ cx: 18, cy: 0, scale: 4 });
  const [hover, setHover] = useState<PickedCell | null>(null);
  const [selected, setSelected] = useState<PickedCell | null>(null);
  const [showBench, setShowBench] = useState(false);

  useEffect(() => {
    api.config().then(setConfig).catch(() => {});
  }, [connected, frame?.header.status.config_version]);

  const applyConfig = useCallback(async (patch: unknown) => {
    setCfgError(null);
    try {
      setConfig(await api.patchConfig(patch));
    } catch (e) {
      setCfgError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const onScrub = useCallback((id: number) => requestFrame(id), [requestFrame]);
  const uniformCell = config?.map.uniform_cell_m ?? 0.05;
  const inspected = hover ?? selected;
  const m = frame?.header.metrics;

  const focus = (x: number, y: number) => {
    setTab("map");
    setView({ cx: x, cy: y, scale: 40 });
  };

  const mapMode = display.baseline;
  return (
    <div className="flex h-full flex-col">
      <Header frame={frame} connected={connected} />
      {frame?.header.status.error && (
        <div className="border-b border-bad/40 bg-bad/10 px-4 py-1.5 text-[12px] text-bad">Engine error: {frame.header.status.error}</div>
      )}
      <div className="flex min-h-0 flex-1">
        <ControlPanel frame={frame} config={config} applyConfig={applyConfig} display={display} setDisplay={setDisplay} onScrub={onScrub} error={cfgError} />

        <main className="flex min-w-0 flex-1 flex-col gap-2 p-2">
          <KpiStrip frame={frame} />
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-line bg-panel">
            <div className="scroll-thin flex items-center overflow-x-auto whitespace-nowrap border-b border-line px-1.5">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`flex shrink-0 items-center gap-1.5 border-b-2 px-3 py-2 text-[12px] font-medium transition-colors ${tab === t.id ? "border-accent text-accent" : "border-transparent text-ink-3 hover:text-ink"}`}
                >
                  {t.icon}
                  {t.label}
                </button>
              ))}
              <button
                onClick={() => setShowBench(true)}
                className="ml-auto mr-1 flex shrink-0 items-center gap-1.5 rounded-md border border-line-2 px-2.5 py-1 text-[12px] text-ink-2 hover:border-accent hover:text-accent"
              >
                <BarChart3 size={14} /> Benchmark report
              </button>
            </div>
            <div className="relative min-h-0 flex-1">
              <div className="absolute inset-0">
              {tab === "map" && (
                <MapCanvas
                  frame={frame}
                  mode={mapMode}
                  display={display}
                  view={view}
                  setView={setView}
                  uniformCell={uniformCell}
                  selected={selected}
                  onHover={setHover}
                  onSelect={setSelected}
                  title={mapMode === "adaptive" ? "Adaptive variable-resolution map" : "Uniform 5 cm baseline"}
                  subtitle={
                    m
                      ? mapMode === "adaptive"
                        ? `${fmtInt(m.occupied_cells)} cells · ${fmtInt(m.points)} points`
                        : `${fmtInt(m.uniform_sparse_cells)} occupied cells (sparse) · ${fmtInt(m.uniform_dense_cells)} if dense`
                      : undefined
                  }
                />
              )}
              {tab === "split" && (
                <div className="grid h-full grid-cols-2 divide-x divide-line">
                  <MapCanvas
                    frame={frame}
                    mode="adaptive"
                    display={display}
                    view={view}
                    setView={setView}
                    uniformCell={uniformCell}
                    selected={selected}
                    onHover={setHover}
                    onSelect={setSelected}
                    title="Adaptive (5 → 50 cm)"
                    subtitle={m ? `${fmtInt(m.occupied_cells)} cells · ${m.memory_proxy_mb.toFixed(2)} MB` : undefined}
                  />
                  <MapCanvas
                    frame={frame}
                    mode="uniform"
                    display={{ ...display, showBands: true }}
                    view={view}
                    setView={setView}
                    uniformCell={uniformCell}
                    selected={selected}
                    onHover={setHover}
                    onSelect={setSelected}
                    title={`Uniform ${fmtSize(uniformCell)}`}
                    subtitle={
                      m
                        ? `${fmtInt(m.uniform_sparse_cells)} cells · ${m.uniform_sparse_memory_proxy_mb.toFixed(2)} MB (dense: ${fmtInt(m.uniform_dense_cells)})`
                        : undefined
                    }
                  />
                </div>
              )}
              {tab === "relief" && <ReliefView frame={frame} display={display} />}
              {tab === "points" && <PointCloudView frame={frame} display={display} />}
              {tab === "architecture" && <ArchitectureView frame={frame} />}
              </div>
            </div>
            {(tab === "map" || tab === "split" || tab === "relief") && (
              <div className="flex items-center justify-between border-t border-line px-3 py-1.5">
                <MapLegend display={display} frame={frame} />
                <span className="text-[10.5px] text-ink-3">scroll = zoom · drag = pan · click = pin cell · double-click = reset</span>
              </div>
            )}
          </div>
          <div className="h-[218px] shrink-0">
            <ChartsPanel series={series} frame={frame} />
          </div>
        </main>

        <RightPanel frame={frame} cell={inspected} pinned={!hover && !!selected} onFocus={focus} />
      </div>
      {showBench && <BenchmarkModal onClose={() => setShowBench(false)} />}
    </div>
  );
}
