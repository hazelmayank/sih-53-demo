// Client-side mirror of the backend resolution schedule, used only for picking and
// overlays. The authoritative cell assignment is always computed by the backend.
import type { Frame, PickedCell, Schedule } from "../types";

export function levelForRange(schedule: Schedule, r: number): number {
  const lv = schedule.levels;
  for (let i = 0; i < lv.length; i++) if (r < lv[i].upper_m) return i; // half-open [lower, upper)
  return lv.length - 1;
}

export function tileLevel(schedule: Schedule, tx: number, ty: number): number {
  const t = schedule.tile_m;
  return levelForRange(schedule, Math.hypot((tx + 0.5) * t, (ty + 0.5) * t));
}

const key = (level: number, ix: number, iy: number) => level * 4294967296 + (ix + 32768) * 65536 + (iy + 32768);

const indexCache = new WeakMap<object, Map<number, number>>();

function cellIndex(frame: Frame, uniform: boolean): Map<number, number> {
  const src = uniform ? frame.uniform! : frame.cells;
  let m = indexCache.get(src);
  if (m) return m;
  m = new Map();
  if (uniform) {
    const u = frame.uniform!;
    for (let i = 0; i < u.ix.length; i++) m.set(key(0, u.ix[i], u.iy[i]), i);
  } else {
    const c = frame.cells;
    for (let i = 0; i < c.ix.length; i++) m.set(key(c.level[i], c.ix[i], c.iy[i]), i);
  }
  indexCache.set(src, m);
  return m;
}

/** Find the allocated cell containing world point (x, y), or null. */
export function pickCell(frame: Frame, x: number, y: number, mode: "adaptive" | "uniform", uniformCell: number): PickedCell | null {
  const s = frame.header.schedule;
  const r = Math.hypot(x, y);
  if (r > s.max_range_m) return null;
  if (mode === "uniform") {
    if (!frame.uniform) return null;
    const ix = Math.floor(x / uniformCell);
    const iy = Math.floor(y / uniformCell);
    const i = cellIndex(frame, true).get(key(0, ix, iy));
    if (i === undefined) return null;
    const u = frame.uniform;
    const cx = (ix + 0.5) * uniformCell;
    const cy = (iy + 0.5) * uniformCell;
    return {
      mode, index: i, level: 0, cellSize: uniformCell, ix, iy, cx, cy, range: Math.hypot(cx, cy),
      zmax: u.zmax[i] / 100, label: u.label[i], frameId: frame.header.frame_id,
    };
  }
  const level = tileLevel(s, Math.floor(x / s.tile_m), Math.floor(y / s.tile_m));
  const size = s.levels[level].cell_size_m;
  const ix = Math.floor(x / size);
  const iy = Math.floor(y / size);
  const i = cellIndex(frame, false).get(key(level, ix, iy));
  if (i === undefined) return null;
  const c = frame.cells;
  const cx = (ix + 0.5) * size;
  const cy = (iy + 0.5) * size;
  return {
    mode, index: i, level, cellSize: size, ix, iy, cx, cy, range: Math.hypot(cx, cy),
    count: c.count[i], zmin: c.zmin[i] / 100, zmax: c.zmax[i] / 100, zmean: c.zmean[i] / 100,
    rough: c.rough[i] / 1000, label: c.label[i], conf: c.conf[i] / 255, dyn: c.dyn[i] / 255, occ: c.occ[i] / 255,
    frameId: frame.header.frame_id,
  };
}

export function fmtInt(n: number | undefined | null): string {
  if (n === undefined || n === null || !Number.isFinite(n)) return "–";
  return Math.round(n).toLocaleString("en-US");
}

export function fmtCompact(n: number | undefined | null): string {
  if (n === undefined || n === null || !Number.isFinite(n)) return "–";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 1 : 2) + "M";
  if (Math.abs(n) >= 1e4) return (n / 1e3).toFixed(1) + "k";
  return Math.round(n).toLocaleString("en-US");
}

export function fmtSize(m: number): string {
  return m < 1 ? `${Math.round(m * 100)} cm` : `${m.toFixed(2)} m`;
}
