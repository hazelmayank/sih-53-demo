"""Binary WebSocket frame protocol.

Frame message layout (little endian)::

    uint32  header_length
    bytes   header (UTF-8 JSON)
    ...     padding to an 8-byte boundary
    ...     sections, each 8-byte aligned; offsets/dtypes are listed in header["sections"]

Sending typed arrays instead of JSON objects keeps a ~50k-cell map at a few
hundred kB per frame, so the dashboard can render every processed frame.

Cell sections (``c_*``, one entry per allocated adaptive cell):
    c_level u8, c_label u8 (dominant class), c_ix i16, c_iy i16, c_count u16,
    c_zmin/c_zmax/c_zmean i16 (cm), c_rough u16 (mm), c_conf/c_dyn/c_occ u8 (x255)
Point sections (``p_*``, optional, sampled): p_x/p_y/p_z i16 (cm), p_label u8, p_int u8
Uniform baseline (``u_*``, optional): u_ix/u_iy i16, u_label u8, u_zmax i16 (cm)
"""

from __future__ import annotations

import json
import struct

import numpy as np

from backend.core.pipeline import ProcessedFrame

_DTYPE = {np.dtype(np.uint8): "u8", np.dtype(np.int16): "i16", np.dtype(np.uint16): "u16",
          np.dtype(np.int32): "i32", np.dtype(np.float32): "f32"}


def _cm(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a * 100.0), -32767, 32767).astype(np.int16)


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


def _sanitize(obj):
    """Make NumPy scalars / NaN JSON-safe."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else round(f, 5)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def encode_frame(pf: ProcessedFrame, options: dict, status: dict) -> bytes:
    c = pf.cells
    sections: list[tuple[str, np.ndarray]] = [
        ("c_level", c.level.astype(np.uint8)),
        ("c_label", c.dominant_label.astype(np.uint8)),
        ("c_ix", c.ix.astype(np.int16)),
        ("c_iy", c.iy.astype(np.int16)),
        ("c_count", np.minimum(c.point_count, 65535).astype(np.uint16)),
        ("c_zmin", _cm(c.min_z)),
        ("c_zmax", _cm(c.max_z)),
        ("c_zmean", _cm(c.mean_z)),
        ("c_rough", np.clip(np.rint(c.roughness * 1000.0), 0, 65535).astype(np.uint16)),
        ("c_conf", _u8(c.confidence)),
        ("c_dyn", _u8(c.dynamic_score)),
        ("c_occ", _u8(c.occupancy)),
    ]
    n_points = 0
    if options.get("points", True) and pf.stream_points:
        sp = pf.stream_points
        n_all = sp["p_x"].shape[0]
        k = min(n_all, int(options.get("max_points", 60000)))
        idx = slice(None) if k == n_all else np.linspace(0, n_all - 1, k).astype(np.int64)
        for name in ("p_x", "p_y", "p_z", "p_label", "p_int"):
            sections.append((name, sp[name][idx]))
        n_points = k
    n_uniform = 0
    if options.get("uniform", False):
        u = pf.uniform
        sections += [("u_ix", u.ix), ("u_iy", u.iy), ("u_label", u.label.astype(np.uint8)), ("u_zmax", _cm(u.max_z))]
        n_uniform = len(u)

    header = pf.summary()
    header.update({
        "type": "frame",
        "counts": {"cells": len(c), "points": n_points, "uniform": n_uniform},
        "status": status,
    })
    # sections are laid out after the header; offsets are relative to the payload start
    layout = []
    offset = 0
    blobs = []
    for name, arr in sections:
        arr = np.ascontiguousarray(arr)
        b = arr.tobytes()
        layout.append({"name": name, "dtype": _DTYPE[arr.dtype], "offset": offset, "length": int(arr.shape[0])})
        blobs.append(b)
        pad = (-len(b)) % 8
        if pad:
            blobs.append(b"\0" * pad)
        offset += len(b) + pad
    header["sections"] = layout
    hbytes = json.dumps(_sanitize(header), separators=(",", ":")).encode("utf-8")
    hpad = (-(4 + len(hbytes))) % 8
    return b"".join([struct.pack("<I", len(hbytes)), hbytes, b" " * hpad, *blobs])
