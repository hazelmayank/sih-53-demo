import { useCallback, useEffect, useRef, useState } from "react";
import { decodeFrame } from "../lib/protocol";
import type { Frame } from "../types";

export interface SeriesPoint {
  frame: number;
  latency: number;
  total: number;
  fps: number;
  adaptiveMs: number;
  uniformMs: number;
  adaptiveCells: number;
  uniformCells: number;
  points: number;
  memAdaptive: number;
  memUniform: number;
  dynamic: number;
  kb: number;
}

export interface StreamOptions {
  points: boolean;
  uniform: boolean;
  max_points: number;
}

const SERIES_LEN = 150;

export function useLidarStream(options: StreamOptions) {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [connected, setConnected] = useState(false);
  const [series, setSeries] = useState<SeriesPoint[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const optsRef = useRef(options);
  const lastIdRef = useRef<number>(-1);
  optsRef.current = options;

  useEffect(() => {
    let closed = false;
    let retry = 0;
    let timer: number | undefined;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => {
        retry = 0;
        setConnected(true);
        ws.send(JSON.stringify({ type: "subscribe", ...optsRef.current }));
      };
      ws.onmessage = (ev) => {
        if (!(ev.data instanceof ArrayBuffer)) return;
        let f: Frame;
        try {
          f = decodeFrame(ev.data);
        } catch (err) {
          console.error("bad frame", err);
          return;
        }
        setFrame(f);
        const m = f.header.metrics;
        const id = f.header.frame_id;
        if (id !== lastIdRef.current && f.header.status.running) {
          lastIdRef.current = id;
          setSeries((prev) => {
            const next = prev.length && prev[prev.length - 1].frame > id ? [] : prev.slice(-SERIES_LEN + 1);
            next.push({
              frame: id,
              latency: m.latency_ms,
              total: m.total_ms,
              fps: m.fps,
              adaptiveMs: m.adaptive_projection_ms,
              uniformMs: m.uniform_projection_ms,
              adaptiveCells: m.occupied_cells,
              uniformCells: m.uniform_sparse_cells,
              points: m.points,
              memAdaptive: m.memory_proxy_mb,
              memUniform: m.uniform_sparse_memory_proxy_mb,
              dynamic: m.dynamic_objects,
              kb: f.bytes / 1024,
            });
            return next;
          });
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (closed) return;
        retry = Math.min(retry + 1, 6);
        timer = window.setTimeout(connect, 300 * 2 ** retry);
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      wsRef.current?.close();
    };
  }, []);

  useEffect(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "subscribe", ...options }));
  }, [options.points, options.uniform, options.max_points]); // eslint-disable-line react-hooks/exhaustive-deps

  const requestFrame = useCallback((frameId: number) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "get_frame", frame_id: frameId }));
  }, []);

  const clearSeries = useCallback(() => setSeries([]), []);

  return { frame, connected, series, requestFrame, clearSeries };
}
