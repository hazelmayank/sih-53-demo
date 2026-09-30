// Colour helpers shared by the 2D canvas and the Three.js views.

export const CLASS_COLORS = ["#2bb39a", "#b0803f", "#f2a93b", "#4ea8ff", "#ff4f8b", "#8b95a5"];
export const CLASS_NAMES = ["Drivable", "Non-drivable", "Static obstacle", "Vehicle", "Pedestrian", "Unknown"];
export const CLASS_SHORT = ["DRV", "NDR", "STA", "VEH", "PED", "UNK"];
export const LEVEL_COLORS = ["#22d3ee", "#a78bfa", "#f472b6", "#facc15", "#34d399", "#fb923c", "#60a5fa", "#e879f9"];

export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// Perceptually ordered ramp (dark blue -> teal -> yellow -> white) for elevation / confidence.
const RAMP: [number, number, number][] = [
  [30, 42, 90],
  [24, 98, 150],
  [23, 158, 160],
  [110, 200, 110],
  [230, 215, 80],
  [250, 245, 220],
];

export function ramp(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  const f = x - i;
  const a = RAMP[i];
  const b = RAMP[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export const rgbCss = (c: [number, number, number]) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;

// Elevation colour range (metres) used by all views.
export const Z_MIN = -1.0;
export const Z_MAX = 5.0;
export const zNorm = (z: number) => (z - Z_MIN) / (Z_MAX - Z_MIN);
