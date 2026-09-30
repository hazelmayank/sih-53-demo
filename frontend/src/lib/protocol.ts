// Decoder for the binary frame protocol (see backend/api/protocol.py).
import type { Frame, FrameHeader } from "../types";

type Section = { name: string; dtype: "u8" | "i16" | "u16" | "i32" | "f32"; offset: number; length: number };

const CTORS = {
  u8: Uint8Array,
  i16: Int16Array,
  u16: Uint16Array,
  i32: Int32Array,
  f32: Float32Array,
} as const;

const decoder = new TextDecoder();

export function decodeFrame(buf: ArrayBuffer): Frame {
  const view = new DataView(buf);
  const hlen = view.getUint32(0, true);
  const header = JSON.parse(decoder.decode(new Uint8Array(buf, 4, hlen))) as FrameHeader & { sections: Section[] };
  const base = Math.ceil((4 + hlen) / 8) * 8;
  const arr: Record<string, ArrayLike<number>> = {};
  for (const s of header.sections) {
    const Ctor = CTORS[s.dtype] as unknown as new (b: ArrayBuffer, offset: number, length: number) => ArrayLike<number>;
    arr[s.name] = new Ctor(buf, base + s.offset, s.length);
  }
  const a = arr as Record<string, never>;
  return {
    header,
    cells: {
      level: a.c_level,
      label: a.c_label,
      ix: a.c_ix,
      iy: a.c_iy,
      count: a.c_count,
      zmin: a.c_zmin,
      zmax: a.c_zmax,
      zmean: a.c_zmean,
      rough: a.c_rough,
      conf: a.c_conf,
      dyn: a.c_dyn,
      occ: a.c_occ,
    },
    points: arr.p_x ? { x: a.p_x, y: a.p_y, z: a.p_z, label: a.p_label, intensity: a.p_int } : null,
    uniform: arr.u_ix ? { ix: a.u_ix, iy: a.u_iy, label: a.u_label, zmax: a.u_zmax } : null,
    bytes: buf.byteLength,
    receivedAt: performance.now(),
  };
}
