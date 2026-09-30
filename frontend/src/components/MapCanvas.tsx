import { useCallback, useEffect, useRef, useState } from "react";
import type { DisplayOptions, Frame, PickedCell } from "../types";
import { CLASS_COLORS, LEVEL_COLORS, hexToRgb, ramp, rgbCss, zNorm } from "../lib/colors";
import { fmtSize, pickCell, tileLevel } from "../lib/grid";

export interface ViewState {
  cx: number; // world x at canvas centre
  cy: number; // world y at canvas centre
  scale: number; // pixels per metre
}

interface Props {
  frame: Frame | null;
  mode: "adaptive" | "uniform";
  display: DisplayOptions;
  view: ViewState;
  setView: (v: ViewState) => void;
  uniformCell: number;
  selected: PickedCell | null;
  onHover: (c: PickedCell | null) => void;
  onSelect: (c: PickedCell | null) => void;
  title?: string;
  subtitle?: string;
}

const CLASS_RGB = CLASS_COLORS.map(hexToRgb);
const N_BUCKETS = 24;

function bucketColors(display: DisplayOptions): string[] {
  switch (display.colorMode) {
    case "class":
      return CLASS_COLORS;
    case "level":
      return LEVEL_COLORS;
    case "dynamic":
      return Array.from({ length: N_BUCKETS }, (_, i) => {
        const t = i / (N_BUCKETS - 1);
        return t < 0.05 ? "#1d2735" : rgbCss([80 + 175 * t, 60 + 30 * t, 200 - 120 * t]);
      });
    default:
      return Array.from({ length: N_BUCKETS }, (_, i) => rgbCss(ramp(i / (N_BUCKETS - 1))));
  }
}

export function defaultView(w: number, h: number): ViewState {
  return { cx: 18, cy: 0, scale: Math.max(1.5, Math.min(w, h) / 2 / 75) };
}

export default function MapCanvas({
  frame, mode, display, view, setView, uniformCell, selected, onHover, onSelect, title, subtitle,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [hover, setHover] = useState<PickedCell | null>(null);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const drag = useRef<{ x: number; y: number; view: ViewState; moved: boolean } | null>(null);
  const initialised = useRef(false);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => {
      const w = Math.max(50, e.contentRect.width);
      const h = Math.max(50, e.contentRect.height);
      setSize({ w, h });
      if (!initialised.current) {
        initialised.current = true;
        setView(defaultView(w, h));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [setView]);

  const toWorld = useCallback(
    (sx: number, sy: number) => ({ x: view.cx + (size.h / 2 - sy) / view.scale, y: view.cy - (sx - size.w / 2) / view.scale }),
    [view, size],
  );

  // ------------------------------------------------------------------ drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const W = size.w;
    const H = size.h;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#06090d";
    ctx.fillRect(0, 0, W, H);

    const { cx, cy, scale: s } = view;
    const sxOf = (y: number) => W / 2 - (y - cy) * s;
    const syOf = (x: number) => H / 2 - (x - cx) * s;
    const xMin = cx - H / 2 / s, xMax = cx + H / 2 / s;
    const yMin = cy - W / 2 / s, yMax = cy + W / 2 / s;

    if (!frame) {
      ctx.fillStyle = "#64748b";
      ctx.font = "13px ui-sans-serif, system-ui";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for the processing engine…", W / 2, H / 2);
      return;
    }
    const sched = frame.header.schedule;
    const R = sched.max_range_m;

    // faint band tint + disk
    if (display.showBands) {
      for (let i = sched.levels.length - 1; i >= 0; i--) {
        const lv = sched.levels[i];
        const rr = Math.min(lv.upper_m, R);
        const [r, g, b] = hexToRgb(LEVEL_COLORS[i % LEVEL_COLORS.length]);
        ctx.fillStyle = `rgba(${r},${g},${b},0.035)`;
        ctx.beginPath();
        ctx.arc(sxOf(0), syOf(0), rr * s, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // ---- cells
    const colors = bucketColors(display);
    const vis = display.classVisible;
    const drawCells = (
      n: number,
      level: (i: number) => number,
      ix: ArrayLike<number>,
      iy: ArrayLike<number>,
      label: ArrayLike<number>,
      bucketOf: (i: number) => number,
      sizeOf: (lv: number) => number,
    ) => {
      const nb = colors.length;
      const lists: number[][] = Array.from({ length: nb }, () => []);
      for (let i = 0; i < n; i++) {
        if (!vis[label[i]]) continue;
        const cs = sizeOf(level(i));
        const x0 = ix[i] * cs;
        const y0 = iy[i] * cs;
        if (x0 > xMax || x0 + cs < xMin || y0 > yMax || y0 + cs < yMin) continue;
        lists[bucketOf(i)].push(i);
      }
      // terrain first so obstacles stay on top when cells are sub-pixel
      const order = display.colorMode === "class" ? [0, 1, 5, 2, 3, 4] : Array.from({ length: nb }, (_, i) => i);
      for (const b of order) {
        const list = lists[b];
        if (!list || !list.length) continue;
        ctx.fillStyle = colors[b];
        for (const i of list) {
          const cs = sizeOf(level(i));
          const px = cs * s;
          const inset = px >= 7 ? 0.6 : 0;
          const w = Math.max(px - 2 * inset, 0.9);
          ctx.fillRect(sxOf((iy[i] + 1) * cs) + inset, syOf((ix[i] + 1) * cs) + inset, w, w);
        }
      }
    };

    if (mode === "uniform") {
      const u = frame.uniform;
      if (u) {
        const bucketOf = (i: number) => {
          switch (display.colorMode) {
            case "class":
              return u.label[i];
            case "level":
              return 0;
            case "elevation":
              return Math.round(Math.min(1, Math.max(0, zNorm(u.zmax[i] / 100))) * (N_BUCKETS - 1));
            default:
              return Math.round(0.5 * (N_BUCKETS - 1));
          }
        };
        drawCells(u.ix.length, () => 0, u.ix, u.iy, u.label, bucketOf, () => uniformCell);
      } else {
        ctx.fillStyle = "#94a3b8";
        ctx.font = "12px ui-sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Loading uniform baseline…", W / 2, H / 2);
      }
    } else {
      const c = frame.cells;
      const sizes = sched.levels.map((l) => l.cell_size_m);
      const bucketOf = (i: number) => {
        switch (display.colorMode) {
          case "class":
            return c.label[i];
          case "level":
            return c.level[i] % LEVEL_COLORS.length;
          case "elevation":
            return Math.round(Math.min(1, Math.max(0, zNorm(c.zmax[i] / 100))) * (N_BUCKETS - 1));
          case "confidence":
            return Math.round((c.conf[i] / 255) * (N_BUCKETS - 1));
          case "dynamic":
            return Math.round((c.dyn[i] / 255) * (N_BUCKETS - 1));
        }
      };
      drawCells(c.ix.length, (i) => c.level[i], c.ix, c.iy, c.label, bucketOf, (lv) => sizes[lv]);
    }

    // ---- raw points overlay
    if (display.showPoints && frame.points) {
      const p = frame.points;
      const n = p.x.length;
      ctx.globalAlpha = 0.9;
      let last = -1;
      for (let i = 0; i < n; i++) {
        const lab = p.label[i];
        if (!vis[lab]) continue;
        const x = p.x[i] / 100, y = p.y[i] / 100;
        if (x < xMin || x > xMax || y < yMin || y > yMax) continue;
        if (lab !== last) {
          const [r, g, b] = CLASS_RGB[lab];
          ctx.fillStyle = `rgb(${Math.min(255, r + 70)},${Math.min(255, g + 70)},${Math.min(255, b + 70)})`;
          last = lab;
        }
        ctx.fillRect(sxOf(y) - 0.7, syOf(x) - 0.7, 1.4, 1.4);
      }
      ctx.globalAlpha = 1;
    }

    // ---- alignment-tile staircase (grid debug)
    if (display.showGridDebug && mode === "adaptive") {
      const t = sched.tile_m;
      const tx0 = Math.floor(xMin / t), tx1 = Math.ceil(xMax / t);
      const ty0 = Math.floor(yMin / t), ty1 = Math.ceil(yMax / t);
      const count = (tx1 - tx0) * (ty1 - ty0);
      if (count < 60000) {
        if (t * s >= 14) {
          ctx.strokeStyle = "rgba(148,163,184,0.18)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let tx = tx0; tx <= tx1; tx++) {
            const yy = syOf(tx * t);
            ctx.moveTo(0, yy);
            ctx.lineTo(W, yy);
          }
          for (let ty = ty0; ty <= ty1; ty++) {
            const xx = sxOf(ty * t);
            ctx.moveTo(xx, 0);
            ctx.lineTo(xx, H);
          }
          ctx.stroke();
        }
        ctx.strokeStyle = "rgba(255,255,255,0.85)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let tx = tx0; tx < tx1; tx++) {
          for (let ty = ty0; ty < ty1; ty++) {
            const L = tileLevel(sched, tx, ty);
            if (tileLevel(sched, tx + 1, ty) !== L) {
              const yy = syOf((tx + 1) * t);
              ctx.moveTo(sxOf(ty * t), yy);
              ctx.lineTo(sxOf((ty + 1) * t), yy);
            }
            if (tileLevel(sched, tx, ty + 1) !== L) {
              const xx = sxOf((ty + 1) * t);
              ctx.moveTo(xx, syOf(tx * t));
              ctx.lineTo(xx, syOf((tx + 1) * t));
            }
          }
        }
        ctx.stroke();
      }
    }

    // ---- band rings + labels
    if (display.showBands) {
      ctx.setLineDash([6, 5]);
      ctx.lineWidth = 1.2;
      ctx.font = "600 11px ui-monospace, Consolas, monospace";
      ctx.textAlign = "left";
      const rings = sched.levels.map((l, i) => ({ r: Math.min(l.upper_m, R), i }));
      for (const { r, i } of rings) {
        const col = LEVEL_COLORS[i % LEVEL_COLORS.length];
        ctx.strokeStyle = col;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.arc(sxOf(0), syOf(0), r * s, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
        const lv = sched.levels[i];
        const mid = (lv.lower_m + Math.min(lv.upper_m, R)) / 2;
        const ang = (-50 * Math.PI) / 180;
        const lx = sxOf(0) + Math.cos(ang) * mid * s;
        const ly = syOf(0) + Math.sin(ang) * mid * s;
        if (lx > 0 && lx < W - 40 && ly > 10 && ly < H) {
          const txt = `${lv.lower_m}–${Math.min(lv.upper_m, R)} m · ${fmtSize(lv.cell_size_m)}`;
          const tw = ctx.measureText(txt).width;
          ctx.fillStyle = "rgba(6,9,13,0.8)";
          ctx.fillRect(lx - 4, ly - 12, tw + 8, 17);
          ctx.fillStyle = col;
          ctx.fillText(txt, lx, ly);
        }
      }
      ctx.setLineDash([]);
    }

    // ---- objects, tracks, velocity vectors
    if (display.showObjects) {
      for (const o of frame.header.objects) {
        if (!vis[o.class]) continue;
        if (o.class === 2 && o.length > 14) continue; // skip long walls to keep the overlay readable
        const [r, g, b] = CLASS_RGB[o.class];
        const ca = Math.cos(o.yaw), sa = Math.sin(o.yaw);
        const hl = Math.max(o.length, 0.3) / 2, hw = Math.max(o.width, 0.3) / 2;
        const pts = [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]].map(([u, v]) => [o.x + u * ca - v * sa, o.y + u * sa + v * ca]);
        if (pts.every(([x, y]) => x < xMin || x > xMax || y < yMin || y > yMax)) continue;
        ctx.strokeStyle = `rgba(${r},${g},${b},${o.class >= 3 ? 1 : 0.55})`;
        ctx.lineWidth = o.dynamic ? 2 : 1.2;
        ctx.beginPath();
        pts.forEach(([x, y], k) => (k ? ctx.lineTo(sxOf(y), syOf(x)) : ctx.moveTo(sxOf(y), syOf(x))));
        ctx.closePath();
        ctx.stroke();
        if (display.showTracks && o.trail.length > 1 && o.dynamic) {
          ctx.strokeStyle = `rgba(${r},${g},${b},0.55)`;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          o.trail.forEach(([x, y], k) => (k ? ctx.lineTo(sxOf(y), syOf(x)) : ctx.moveTo(sxOf(y), syOf(x))));
          ctx.stroke();
          ctx.setLineDash([]);
        }
        if (display.showTracks && o.dynamic && o.speed > 0.2) {
          const ex = o.x + o.vx * 1.0, ey = o.y + o.vy * 1.0; // 1 s look-ahead
          const x0 = sxOf(o.y), y0 = syOf(o.x), x1 = sxOf(ey), y1 = syOf(ex);
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          const a = Math.atan2(y1 - y0, x1 - x0);
          ctx.lineTo(x1 - 7 * Math.cos(a - 0.45), y1 - 7 * Math.sin(a - 0.45));
          ctx.moveTo(x1, y1);
          ctx.lineTo(x1 - 7 * Math.cos(a + 0.45), y1 - 7 * Math.sin(a + 0.45));
          ctx.stroke();
        }
        if ((o.class === 3 || o.class === 4) && s > 3) {
          ctx.font = "600 10px ui-monospace, Consolas, monospace";
          ctx.textAlign = "left";
          const txt = `${o.class === 3 ? "V" : "P"}${o.track_id ?? "?"}${o.dynamic ? ` ${o.speed.toFixed(1)}m/s` : ""}`;
          const lx = sxOf(o.y) + hw * s + 4, ly = syOf(o.x) - 4;
          ctx.fillStyle = "rgba(6,9,13,0.75)";
          ctx.fillRect(lx - 2, ly - 10, ctx.measureText(txt).width + 4, 13);
          ctx.fillStyle = o.dynamic ? "#ffffff" : `rgb(${r},${g},${b})`;
          ctx.fillText(txt, lx, ly);
        }
      }
    }

    // ---- ego vehicle + sensor
    const ex = sxOf(0), ey = syOf(0);
    ctx.save();
    ctx.translate(ex, ey);
    const carL = 4.6 * s, carW = 1.9 * s;
    ctx.fillStyle = "rgba(34,211,238,0.18)";
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1.5;
    if (carL > 10) {
      ctx.beginPath();
      ctx.roundRect(-carW / 2, -carL * 0.62, carW, carL, 3);
      ctx.fill();
      ctx.stroke();
    }
    ctx.fillStyle = "#22d3ee";
    ctx.beginPath();
    ctx.moveTo(0, -9);
    ctx.lineTo(6, 6);
    ctx.lineTo(0, 3);
    ctx.lineTo(-6, 6);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // ---- hover + selection outlines
    const outline = (c: PickedCell | null, color: string, width: number) => {
      if (!c || c.mode !== mode) return;
      const x0 = c.ix * c.cellSize, y0 = c.iy * c.cellSize;
      const px = Math.max(c.cellSize * s, 4);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.strokeRect(sxOf(y0 + c.cellSize), syOf(x0 + c.cellSize), px, px);
    };
    outline(hover, "#ffffff", 1.5);
    outline(selected, "#22d3ee", 2.5);

    // ---- scale bar
    const nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
    const target = 110 / s;
    const len = nice.find((v) => v >= target) ?? 100;
    ctx.strokeStyle = "#9aa9bd";
    ctx.fillStyle = "#9aa9bd";
    ctx.lineWidth = 2;
    const bx = 14, by = H - 16;
    ctx.beginPath();
    ctx.moveTo(bx, by - 5);
    ctx.lineTo(bx, by);
    ctx.lineTo(bx + len * s, by);
    ctx.lineTo(bx + len * s, by - 5);
    ctx.stroke();
    ctx.font = "11px ui-monospace, Consolas, monospace";
    ctx.textAlign = "left";
    ctx.fillText(len >= 1 ? `${len} m` : `${len * 100} cm`, bx + len * s + 6, by);
  }, [frame, mode, display, view, size, hover, selected, uniformCell]);

  // ------------------------------------------------------------------ interaction
  const onWheel = (e: React.WheelEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const w = toWorld(sx, sy);
    const scale = Math.min(800, Math.max(0.8, view.scale * Math.exp(-e.deltaY * 0.0015)));
    setView({ scale, cx: w.x - (size.h / 2 - sy) / scale, cy: w.y + (sx - size.w / 2) / scale });
  };

  const onMouseDown = (e: React.MouseEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, view, moved: false };
  };

  const onMouseMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const d = drag.current;
    if (d) {
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
      if (d.moved) {
        setView({ scale: d.view.scale, cx: d.view.cx + dy / d.view.scale, cy: d.view.cy + dx / d.view.scale });
        return;
      }
    }
    const w = toWorld(sx, sy);
    setCursor(w);
    if (frame) {
      const c = pickCell(frame, w.x, w.y, mode, uniformCell);
      setHover(c);
      onHover(c);
    }
  };

  const onMouseUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved) onSelect(hover);
  };

  const onLeave = () => {
    drag.current = null;
    setHover(null);
    setCursor(null);
    onHover(null);
  };

  const zoomTo = (radius: number, cx = 0) => setView({ cx, cy: 0, scale: Math.min(size.w, size.h) / 2 / radius });

  const curLevel =
    cursor && frame && mode === "adaptive"
      ? frame.header.schedule.levels[tileLevel(frame.header.schedule, Math.floor(cursor.x / frame.header.schedule.tile_m), Math.floor(cursor.y / frame.header.schedule.tile_m))]
      : null;

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden select-none">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 block"
        style={{ width: size.w, height: size.h, cursor: drag.current?.moved ? "grabbing" : "crosshair" }}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onLeave}
        onDoubleClick={() => setView(defaultView(size.w, size.h))}
      />
      {title && (
        <div className="pointer-events-none absolute left-3 top-2.5 rounded-md border border-line bg-bg/80 px-2.5 py-1.5">
          <div className="text-[12px] font-semibold text-ink">{title}</div>
          {subtitle && <div className="num text-[11px] text-ink-2">{subtitle}</div>}
        </div>
      )}
      <div className="absolute right-2.5 top-2.5 flex gap-1">
        {[
          ["10 m", 11, 3],
          ["30 m", 32, 8],
          ["60 m", 62, 15],
          ["100 m", 104, 18],
        ].map(([label, r, c]) => (
          <button
            key={label as string}
            onClick={() => zoomTo(r as number, c as number)}
            className="rounded border border-line-2 bg-panel/90 px-2 py-0.5 text-[11px] text-ink-2 hover:border-accent hover:text-accent"
          >
            {label}
          </button>
        ))}
      </div>
      {cursor && (
        <div className="num pointer-events-none absolute bottom-2 right-2.5 rounded bg-bg/80 px-2 py-1 text-[11px] text-ink-2">
          x {cursor.x.toFixed(2)} y {cursor.y.toFixed(2)} r {Math.hypot(cursor.x, cursor.y).toFixed(2)} m
          {curLevel && (
            <span className="ml-2" style={{ color: LEVEL_COLORS[curLevel.level % LEVEL_COLORS.length] }}>
              L{curLevel.level} · {fmtSize(curLevel.cell_size_m)} cells
            </span>
          )}
          {mode === "uniform" && <span className="ml-2 text-ink-3">{fmtSize(uniformCell)} cells</span>}
        </div>
      )}
    </div>
  );
}
