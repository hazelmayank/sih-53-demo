import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { LEVEL_COLORS } from "../lib/colors";
import type { Schedule } from "../types";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // z-up world, matching the vehicle frame

export type CameraPreset = "orbit" | "top" | "chase";

function CameraRig({ preset, nonce }: { preset: CameraPreset; nonce: number }) {
  const { camera } = useThree();
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null;
  useEffect(() => {
    const target = new THREE.Vector3(12, 0, 0);
    if (preset === "top") {
      camera.position.set(11.99, 0, 120); // tiny x offset keeps screen-up = +x (forward)
      target.set(12, 0, 0);
    } else if (preset === "chase") {
      camera.position.set(-14, 0, 7);
      target.set(14, 0, 0);
    } else {
      camera.position.set(-30, 28, 34);
    }
    camera.up.set(0, 0, 1);
    camera.lookAt(target);
    if (controls) {
      controls.target.copy(target);
      controls.update();
    }
  }, [preset, nonce, camera, controls]);
  return null;
}

function Rings({ schedule }: { schedule: Schedule | null }) {
  const lines = useMemo(() => {
    if (!schedule) return [];
    return schedule.levels.map((l, i) => {
      const r = Math.min(l.upper_m, schedule.max_range_m);
      const pts: number[] = [];
      for (let k = 0; k <= 180; k++) {
        const a = (k / 180) * Math.PI * 2;
        pts.push(Math.cos(a) * r, Math.sin(a) * r, 0.02);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      return { g, color: LEVEL_COLORS[i % LEVEL_COLORS.length] };
    });
  }, [schedule]);
  return (
    <>
      {lines.map(({ g, color }, i) => (
        <lineLoop key={i} geometry={g}>
          <lineBasicMaterial color={color} transparent opacity={0.7} />
        </lineLoop>
      ))}
    </>
  );
}

function Ego({ height }: { height: number }) {
  return (
    <group>
      <mesh position={[-0.55, 0, 0.8]}>
        <boxGeometry args={[4.6, 1.9, 1.4]} />
        <meshStandardMaterial color="#22d3ee" transparent opacity={0.25} />
      </mesh>
      <mesh position={[0, 0, height]}>
        <cylinderGeometry args={[0.12, 0.12, 0.15, 16]} />
        <meshBasicMaterial color="#22d3ee" />
      </mesh>
    </group>
  );
}

export default function Scene3D({
  children,
  schedule,
  preset,
  nonce,
  sensorHeight,
}: {
  children: ReactNode;
  schedule: Schedule | null;
  preset: CameraPreset;
  nonce: number;
  sensorHeight: number;
}) {
  const ref = useRef<OrbitControlsImpl>(null);
  return (
    <Canvas
      camera={{ fov: 50, near: 0.1, far: 2000, position: [-30, 28, 34], up: [0, 0, 1] }}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      dpr={[1, 2]}
      style={{ background: "#06090d" }}
    >
      <ambientLight intensity={0.55} />
      <directionalLight position={[-40, 30, 80]} intensity={1.4} />
      <directionalLight position={[50, -40, 30]} intensity={0.35} />
      <OrbitControls ref={ref} makeDefault enableDamping dampingFactor={0.12} maxDistance={400} />
      <CameraRig preset={preset} nonce={nonce} />
      <Rings schedule={schedule} />
      <Ego height={sensorHeight} />
      {children}
    </Canvas>
  );
}
