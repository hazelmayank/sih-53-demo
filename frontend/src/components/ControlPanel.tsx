import { useEffect, useRef, useState } from "react";
import { Pause, Play, Plus, RotateCcw, SkipForward, Trash2, Upload } from "lucide-react";
import { api } from "../lib/api";
import { CLASS_COLORS, CLASS_NAMES, LEVEL_COLORS } from "../lib/colors";
import { fmtSize } from "../lib/grid";
import type { AppConfig, Band, ColorMode, DisplayOptions, Frame } from "../types";
import { Button, Dot, Section, Segmented, Toggle } from "./ui";

const BAND_PRESETS: { name: string; bands: Band[]; hint: string }[] = [
  { name: "Reference", hint: "PS 26053: 5 / 10 / 25 / 50 cm", bands: [
    { upper_m: 10, cell_size_m: 0.05 }, { upper_m: 30, cell_size_m: 0.1 }, { upper_m: 60, cell_size_m: 0.25 }, { upper_m: 100, cell_size_m: 0.5 }] },
  { name: "Aggressive", hint: "5 / 20 / 50 / 100 cm", bands: [
    { upper_m: 8, cell_size_m: 0.05 }, { upper_m: 25, cell_size_m: 0.2 }, { upper_m: 50, cell_size_m: 0.5 }, { upper_m: 100, cell_size_m: 1.0 }] },
  { name: "Fine", hint: "5 / 5 / 10 / 25 cm", bands: [
    { upper_m: 20, cell_size_m: 0.05 }, { upper_m: 40, cell_size_m: 0.1 }, { upper_m: 70, cell_size_m: 0.1 }, { upper_m: 100, cell_size_m: 0.25 }] },
  { name: "Uniform", hint: "5 cm everywhere (baseline)", bands: [{ upper_m: 100, cell_size_m: 0.05 }] },
];

interface Props {
  frame: Frame | null;
  config: AppConfig | null;
  applyConfig: (patch: unknown) => Promise<void>;
  display: DisplayOptions;
  setDisplay: (d: DisplayOptions) => void;
  onScrub: (frameId: number) => void;
  error: string | null;
}

export default function ControlPanel({ frame, config, applyConfig, display, setDisplay, onScrub, error }: Props) {
  const st = frame?.header.status;
  const running = !!st?.running;
  const [presets, setPresets] = useState<{ id: string; label: string; description: string }[]>([]);
  const [files, setFiles] = useState<{ path: string; kind: string; frames: number }[]>([]);
  const [bands, setBands] = useState<Band[]>([]);
  const [maxRange, setMaxRange] = useState(100);
  const [busy, setBusy] = useState(false);
  const uploadRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.presets().then(setPresets).catch(() => {});
    api.files().then(setFiles).catch(() => {});
  }, []);
  useEffect(() => {
    if (!config) return;
    setBands(config.map.bands.map((b) => ({ ...b })));
    setMaxRange(config.processing.max_range_m);
  }, [config]);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };

  const bandsDirty = config && JSON.stringify(bands) !== JSON.stringify(config.map.bands);
  const history = st?.history ?? [];
  const set = (patch: Partial<DisplayOptions>) => setDisplay({ ...display, ...patch });

  return (
    <aside className="scroll-thin flex w-[292px] shrink-0 flex-col overflow-y-auto border-r border-line bg-panel">
      {/* ------------------------------------------------ playback */}
      <Section title="Playback">
        <div className="flex gap-1.5">
          {running ? (
            <Button variant="primary" className="flex-1" onClick={() => act(api.stop)}>
              <Pause size={14} /> Pause
            </Button>
          ) : (
            <Button variant="primary" className="flex-1" onClick={() => act(api.start)}>
              <Play size={14} /> Start
            </Button>
          )}
          <Button onClick={() => act(api.step)} disabled={running} title="Process one frame">
            <SkipForward size={14} />
          </Button>
          <Button onClick={() => act(api.reset)} title="Reset to frame 0">
            <RotateCcw size={14} />
          </Button>
        </div>
        <div className="mt-2.5 flex items-center gap-2 text-[11.5px] text-ink-3">
          <span className="w-12">Speed</span>
          <Segmented
            size="xs"
            value={String(config?.stream.playback_speed ?? 1)}
            options={["0.25", "0.5", "1", "2"].map((v) => ({ value: v, label: `${v}×` }))}
            onChange={(v) => applyConfig({ stream: { playback_speed: Number(v) } })}
          />
        </div>
        <div className="mt-2.5">
          <div className="flex justify-between text-[11px] text-ink-3">
            <span>Frame history {running ? "(pause to scrub)" : ""}</span>
            <span className="num text-ink-2">#{frame?.header.frame_id ?? "–"}</span>
          </div>
          <input
            type="range"
            className="w-full"
            disabled={running || history.length < 2}
            min={history[0] ?? 0}
            max={history[history.length - 1] ?? 0}
            value={frame?.header.frame_id ?? 0}
            onChange={(e) => onScrub(Number(e.target.value))}
          />
        </div>
      </Section>

      {/* ------------------------------------------------ scene */}
      <Section title="Scene / sensor">
        <Segmented
          value={config?.scene.source ?? "synthetic"}
          options={[
            { value: "synthetic", label: "Synthetic LiDAR" },
            { value: "file", label: "Imported file" },
          ]}
          onChange={(v) => {
            if (v === "synthetic") applyConfig({ scene: { source: "synthetic" } });
            else if (files.length) applyConfig({ scene: { source: "file", file_sequence: config?.scene.file_sequence ?? files[0].path } });
            else uploadRef.current?.click();
          }}
        />
        {config?.scene.source !== "file" ? (
          <div className="mt-2 grid grid-cols-2 gap-1.5">
            {presets.map((p) => (
              <button
                key={p.id}
                title={p.description}
                onClick={() => applyConfig({ scene: { preset: p.id } })}
                className={`rounded-md border px-2 py-1.5 text-left text-[11.5px] leading-tight transition-colors ${config?.scene.preset === p.id ? "border-accent bg-accent/10 text-accent" : "border-line-2 text-ink-2 hover:text-ink"}`}
              >
                {p.label}
              </button>
            ))}
          </div>
        ) : (
          <div className="mt-2 space-y-1.5">
            <select
              className="w-full"
              value={config?.scene.file_sequence ?? ""}
              onChange={(e) => applyConfig({ scene: { source: "file", file_sequence: e.target.value } })}
            >
              {files.map((f) => (
                <option key={f.path} value={f.path}>
                  {f.path} ({f.frames} frame{f.frames > 1 ? "s" : ""})
                </option>
              ))}
            </select>
            <label className="flex items-center justify-between text-[11.5px] text-ink-3">
              z offset (m)
              <input
                type="number"
                step={0.01}
                className="w-20"
                defaultValue={config?.scene.import_z_offset_m ?? 0}
                onBlur={(e) => applyConfig({ scene: { import_z_offset_m: Number(e.target.value) } })}
              />
            </label>
          </div>
        )}
        <div className="mt-2 flex items-center gap-2">
          <Button className="flex-1" onClick={() => uploadRef.current?.click()}>
            <Upload size={13} /> Load .bin / .pcd / .ply
          </Button>
          <input
            ref={uploadRef}
            type="file"
            accept=".bin,.pcd,.ply,.npy,.csv,.txt"
            className="hidden"
            onChange={async (e) => {
              const f = e.target.files?.[0];
              if (!f) return;
              await act(async () => {
                await api.upload(f);
                setFiles(await api.files());
                await applyConfig({});
              });
              e.target.value = "";
            }}
          />
        </div>
        {config?.scene.source !== "file" && (
          <div className="mt-2.5 space-y-1.5 text-[11.5px] text-ink-3">
            <div className="flex items-center justify-between gap-2">
              <span>Beams</span>
              <Segmented
                size="xs"
                value={String(config?.scene.beams ?? 64)}
                options={["32", "64", "128"].map((v) => ({ value: v, label: v }))}
                onChange={(v) => applyConfig({ scene: { beams: Number(v) } })}
              />
            </div>
            <div className="flex items-center justify-between gap-2">
              <span>Seed</span>
              <input
                type="number"
                className="w-20"
                defaultValue={config?.scene.seed ?? 7}
                key={config?.scene.seed}
                onBlur={(e) => applyConfig({ scene: { seed: Number(e.target.value) } })}
              />
            </div>
            <div>
              <div className="flex justify-between">
                <span>Range noise σ</span>
                <span className="num text-ink-2">{((config?.scene.range_noise_m ?? 0) * 100).toFixed(1)} cm</span>
              </div>
              <input
                type="range"
                min={0}
                max={0.1}
                step={0.005}
                className="w-full"
                defaultValue={config?.scene.range_noise_m ?? 0.015}
                key={`n${config?.scene.range_noise_m}`}
                onMouseUp={(e) => applyConfig({ scene: { range_noise_m: Number((e.target as HTMLInputElement).value) } })}
              />
            </div>
          </div>
        )}
      </Section>

      {/* ------------------------------------------------ resolution bands */}
      <Section
        title="Resolution bands"
        right={frame && <span className="num text-[10.5px] text-ink-3">tile {fmtSize(frame.header.schedule.tile_m)}</span>}
      >
        <div className="mb-1.5 grid grid-cols-[14px_1fr_1fr_22px] items-center gap-x-2 text-[10px] uppercase tracking-wider text-ink-3">
          <span />
          <span>up to (m)</span>
          <span>cell (m)</span>
          <span />
        </div>
        {bands.map((b, i) => (
          <div key={i} className="mb-1 grid grid-cols-[14px_1fr_1fr_22px] items-center gap-x-2">
            <Dot color={LEVEL_COLORS[i % LEVEL_COLORS.length]} />
            <input
              type="number"
              className="w-full min-w-0"
              step={1}
              value={b.upper_m}
              onChange={(e) => setBands(bands.map((x, j) => (j === i ? { ...x, upper_m: Number(e.target.value) } : x)))}
            />
            <input
              type="number"
              className="w-full min-w-0"
              step={0.05}
              value={b.cell_size_m}
              onChange={(e) => setBands(bands.map((x, j) => (j === i ? { ...x, cell_size_m: Number(e.target.value) } : x)))}
            />
            <button className="text-ink-3 hover:text-bad disabled:opacity-30" disabled={bands.length < 2} onClick={() => setBands(bands.filter((_, j) => j !== i))}>
              <Trash2 size={13} />
            </button>
          </div>
        ))}
        <div className="mt-1.5 flex gap-1.5">
          <Button
            variant="ghost"
            className="px-1.5"
            disabled={bands.length >= 8}
            onClick={() => {
              const last = bands[bands.length - 1];
              setBands([...bands, { upper_m: (last?.upper_m ?? 0) + 20, cell_size_m: last?.cell_size_m ?? 0.5 }]);
            }}
          >
            <Plus size={13} /> band
          </Button>
          <Button variant="primary" className="ml-auto" disabled={!bandsDirty || busy} onClick={() => act(() => applyConfig({ map: { bands } }))}>
            Apply
          </Button>
        </div>
        <div className="mt-2 grid grid-cols-2 gap-1">
          {BAND_PRESETS.map((p) => (
            <button
              key={p.name}
              title={p.hint}
              onClick={() => {
                setBands(p.bands);
                applyConfig({ map: { bands: p.bands } });
              }}
              className="rounded border border-line-2 px-1.5 py-1 text-[11px] text-ink-2 hover:border-accent hover:text-accent"
            >
              {p.name}
            </button>
          ))}
        </div>
        {error && <div className="mt-2 rounded border border-bad/40 bg-bad/10 px-2 py-1.5 text-[11px] leading-snug text-bad">{error}</div>}
        <div className="mt-3">
          <div className="flex justify-between text-[11.5px] text-ink-3">
            <span>Max range</span>
            <span className="num text-ink-2">{maxRange} m</span>
          </div>
          <input
            type="range"
            min={20}
            max={150}
            step={5}
            className="w-full"
            value={maxRange}
            onChange={(e) => setMaxRange(Number(e.target.value))}
            onMouseUp={() => applyConfig({ processing: { max_range_m: maxRange } })}
            onKeyUp={() => applyConfig({ processing: { max_range_m: maxRange } })}
          />
        </div>
      </Section>

      {/* ------------------------------------------------ display */}
      <Section title="Map display">
        <div className="mb-2">
          <div className="mb-1 text-[11px] text-ink-3">Representation</div>
          <Segmented
            value={display.baseline}
            options={[
              { value: "adaptive", label: "Adaptive" },
              { value: "uniform", label: "Uniform 5 cm" },
            ]}
            onChange={(v) => set({ baseline: v })}
          />
        </div>
        <div className="mb-2">
          <div className="mb-1 text-[11px] text-ink-3">Colour cells by</div>
          <select className="w-full" value={display.colorMode} onChange={(e) => set({ colorMode: e.target.value as ColorMode })}>
            <option value="class">Semantic class (dominant)</option>
            <option value="level">Resolution level</option>
            <option value="elevation">Elevation (max z)</option>
            <option value="confidence">Classification confidence</option>
            <option value="dynamic">Dynamic score</option>
          </select>
        </div>
        <Toggle label="Resolution-band overlay" checked={display.showBands} onChange={(v) => set({ showBands: v })} />
        <Toggle label="Grid debug (tile staircase)" checked={display.showGridDebug} onChange={(v) => set({ showGridDebug: v })} hint="Draws where the level changes between 0.5 m alignment tiles" />
        <Toggle label="Raw point overlay" checked={display.showPoints} onChange={(v) => set({ showPoints: v })} />
        <Toggle label="Objects" checked={display.showObjects} onChange={(v) => set({ showObjects: v })} />
        <Toggle label="Tracks & velocity vectors" checked={display.showTracks} onChange={(v) => set({ showTracks: v })} />
      </Section>

      <Section title="Classes">
        {CLASS_NAMES.map((n, i) => (
          <Toggle
            key={n}
            label={
              <span className="flex items-center gap-2">
                <Dot color={CLASS_COLORS[i]} /> {n}
              </span>
            }
            checked={display.classVisible[i]}
            onChange={(v) => set({ classVisible: display.classVisible.map((x, j) => (j === i ? v : x)) })}
          />
        ))}
      </Section>

      <Section title="Segmentation model">
        <select
          className="w-full"
          value={config?.processing.segmenter ?? "geometric_fallback"}
          onChange={(e) => applyConfig({ processing: { segmenter: e.target.value } })}
        >
          <option value="geometric_fallback">Geometric fallback (rule-based)</option>
          <option value="oracle_gt">Ground-truth oracle (synthetic, debug)</option>
          <option value="torchscript" disabled>
            TorchScript PointNet++ / SparseConv (set model_path)
          </option>
        </select>
        <p className="mt-1.5 text-[10.5px] leading-snug text-ink-3">
          The mapping engine only consumes per-point labels + confidences, so a trained network plugs in without any change downstream.
        </p>
      </Section>
    </aside>
  );
}
