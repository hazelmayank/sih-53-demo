import { useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import Scene3D, { type CameraPreset } from "./Scene3D";
import { CLASS_COLORS, LEVEL_COLORS, ramp, zNorm } from "../lib/colors";
import type { DisplayOptions, Frame } from "../types";

const CLASS_C = CLASS_COLORS.map((c) => new THREE.Color(c));
const LEVEL_C = LEVEL_COLORS.map((c) => new THREE.Color(c));

function Columns({ frame, display }: { frame: Frame; display: DisplayOptions }) {
  const n = frame.cells.ix.length;
  const capacity = useMemo(() => Math.max(65536, 2 ** Math.ceil(Math.log2(n + 1))), [n > 65536 ? n : 0]); // eslint-disable-line react-hooks/exhaustive-deps
  const ref = useRef<THREE.InstancedMesh>(null);
  const geom = useMemo(() => new THREE.BoxGeometry(1, 1, 1), []);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const c = frame.cells;
    const sizes = frame.header.schedule.levels.map((l) => l.cell_size_m);
    const m = mesh.instanceMatrix.array as Float32Array;
    if (!mesh.instanceColor) mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    const col = mesh.instanceColor.array as Float32Array;
    const vis = display.classVisible;
    let k = 0;
    const tmp = new THREE.Color();
    for (let i = 0; i < n && k < capacity; i++) {
      const lab = c.label[i];
      if (!vis[lab]) continue;
      const s = sizes[c.level[i]];
      const zmin = c.zmin[i] / 100;
      const zmax = c.zmax[i] / 100;
      const h = Math.max(zmax - zmin, 0.04);
      const w = s * 0.94;
      const o = k * 16;
      m[o] = w; m[o + 1] = 0; m[o + 2] = 0; m[o + 3] = 0;
      m[o + 4] = 0; m[o + 5] = w; m[o + 6] = 0; m[o + 7] = 0;
      m[o + 8] = 0; m[o + 9] = 0; m[o + 10] = h; m[o + 11] = 0;
      m[o + 12] = (c.ix[i] + 0.5) * s; m[o + 13] = (c.iy[i] + 0.5) * s; m[o + 14] = zmin + h / 2; m[o + 15] = 1;
      let color: THREE.Color;
      if (display.colorMode === "level") color = LEVEL_C[c.level[i] % LEVEL_C.length];
      else if (display.colorMode === "elevation") {
        const [r, g, b] = ramp(zNorm(zmax));
        color = tmp.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
      } else if (display.colorMode === "confidence") {
        const [r, g, b] = ramp(c.conf[i] / 255);
        color = tmp.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
      } else if (display.colorMode === "dynamic" && c.dyn[i] > 12) {
        color = tmp.setRGB(1, 0.35, 0.3);
      } else color = CLASS_C[lab];
      col[k * 3] = color.r;
      col[k * 3 + 1] = color.g;
      col[k * 3 + 2] = color.b;
      k++;
    }
    mesh.count = k;
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [frame, display, capacity, n]);

  return (
    <instancedMesh key={capacity} ref={ref} args={[geom, undefined, capacity]} frustumCulled={false}>
      <meshLambertMaterial />
    </instancedMesh>
  );
}

export default function ReliefView({ frame, display }: { frame: Frame | null; display: DisplayOptions }) {
  const [preset, setPreset] = useState<CameraPreset>("orbit");
  const [nonce, setNonce] = useState(0);
  return (
    <div className="relative h-full w-full">
      <Scene3D schedule={frame?.header.schedule ?? null} preset={preset} nonce={nonce} sensorHeight={frame?.header.sensor.z ?? 1.8}>
        {frame && <Columns frame={frame} display={display} />}
      </Scene3D>
      <ViewButtons preset={preset} onPick={(p) => { setPreset(p); setNonce((x) => x + 1); }} />
      <div className="pointer-events-none absolute left-3 top-2.5 max-w-sm rounded-md border border-line bg-bg/80 px-2.5 py-1.5 text-[11.5px] text-ink-2">
        <div className="text-[12px] font-semibold text-ink">2.5D elevation map</div>
        Each allocated cell is a column from <span className="num text-ink">min_z</span> to{" "}
        <span className="num text-ink">max_z</span> with its true footprint: fine near the vehicle, coarse far away,
        with height and overhangs kept.
      </div>
    </div>
  );
}

export function ViewButtons({ preset, onPick }: { preset: CameraPreset; onPick: (p: CameraPreset) => void }) {
  return (
    <div className="absolute right-2.5 top-2.5 flex gap-1">
      {(["orbit", "top", "chase"] as CameraPreset[]).map((p) => (
        <button
          key={p}
          onClick={() => onPick(p)}
          className={`rounded border px-2 py-0.5 text-[11px] capitalize ${preset === p ? "border-accent text-accent" : "border-line-2 bg-panel/90 text-ink-2 hover:text-ink"}`}
        >
          {p === "top" ? "top-down" : p}
        </button>
      ))}
    </div>
  );
}
