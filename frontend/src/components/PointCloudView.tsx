import { useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import Scene3D, { type CameraPreset } from "./Scene3D";
import { ViewButtons } from "./ReliefView";
import { CLASS_COLORS, ramp, zNorm } from "../lib/colors";
import type { DisplayOptions, Frame } from "../types";

const CLASS_C = CLASS_COLORS.map((c) => new THREE.Color(c));
const CAP = 320000;

type PcColor = "class" | "height" | "intensity";

function Cloud({ frame, display, colorBy }: { frame: Frame; display: DisplayOptions; colorBy: PcColor }) {
  const ref = useRef<THREE.Points>(null);
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(CAP * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(CAP * 3), 3));
    return g;
  }, []);

  useLayoutEffect(() => {
    const p = frame.points;
    if (!p) return;
    const pos = geom.getAttribute("position") as THREE.BufferAttribute;
    const col = geom.getAttribute("color") as THREE.BufferAttribute;
    const P = pos.array as Float32Array;
    const Cc = col.array as Float32Array;
    const vis = display.classVisible;
    const tmp = new THREE.Color();
    let k = 0;
    for (let i = 0; i < p.x.length && k < CAP; i++) {
      const lab = p.label[i];
      if (!vis[lab]) continue;
      const z = p.z[i] / 100;
      P[k * 3] = p.x[i] / 100;
      P[k * 3 + 1] = p.y[i] / 100;
      P[k * 3 + 2] = z;
      let c: THREE.Color;
      if (colorBy === "class") c = CLASS_C[lab];
      else {
        const [r, g, b] = ramp(colorBy === "height" ? zNorm(z) : p.intensity[i] / 255);
        c = tmp.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
      }
      Cc[k * 3] = c.r;
      Cc[k * 3 + 1] = c.g;
      Cc[k * 3 + 2] = c.b;
      k++;
    }
    geom.setDrawRange(0, k);
    pos.needsUpdate = true;
    col.needsUpdate = true;
    geom.computeBoundingSphere();
  }, [frame, display, colorBy, geom]);

  return (
    <points ref={ref} geometry={geom} frustumCulled={false}>
      <pointsMaterial size={2.2} sizeAttenuation={false} vertexColors />
    </points>
  );
}

export default function PointCloudView({ frame, display }: { frame: Frame | null; display: DisplayOptions }) {
  const [preset, setPreset] = useState<CameraPreset>("orbit");
  const [nonce, setNonce] = useState(0);
  const [colorBy, setColorBy] = useState<PcColor>("class");
  return (
    <div className="relative h-full w-full">
      <Scene3D schedule={frame?.header.schedule ?? null} preset={preset} nonce={nonce} sensorHeight={frame?.header.sensor.z ?? 1.8}>
        {frame && <Cloud frame={frame} display={display} colorBy={colorBy} />}
      </Scene3D>
      <ViewButtons preset={preset} onPick={(p) => { setPreset(p); setNonce((x) => x + 1); }} />
      <div className="absolute left-3 top-2.5 rounded-md border border-line bg-bg/80 px-2.5 py-1.5">
        <div className="text-[12px] font-semibold text-ink">Raw LiDAR point cloud</div>
        <div className="num text-[11px] text-ink-2">
          showing {frame?.points?.x.length.toLocaleString() ?? 0} of {frame?.header.metrics.points.toLocaleString() ?? 0} processed points
        </div>
        <div className="mt-1 flex gap-1">
          {(["class", "height", "intensity"] as PcColor[]).map((c) => (
            <button
              key={c}
              onClick={() => setColorBy(c)}
              className={`rounded border px-1.5 py-0.5 text-[10.5px] capitalize ${colorBy === c ? "border-accent text-accent" : "border-line-2 text-ink-2"}`}
            >
              {c}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
