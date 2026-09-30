"""Point-cloud file import (Mode B) with pure-NumPy readers.

Supported: KITTI-style ``.bin`` (float32 x, y, z, intensity), ``.pcd``
(ascii / binary; binary_compressed via Open3D if installed), ``.ply``
(ascii / binary_little_endian), ``.npy`` (N x 3+), ``.csv`` / ``.txt``
(x, y, z[, intensity]). A directory is treated as a frame sequence in
sorted filename order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.lidar.frame import PointFrame, validate_points

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
SUPPORTED = {".bin", ".pcd", ".ply", ".npy", ".csv", ".txt"}


def read_points(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"{path.name}: KITTI .bin size is not a multiple of 4 floats")
        pts = raw.reshape(-1, 4)
        return pts[:, :3], pts[:, 3]
    if ext == ".npy":
        arr = np.load(path)
        return arr[:, :3], (arr[:, 3] if arr.shape[1] > 3 else None)
    if ext in (".csv", ".txt"):
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None, ndmin=2, comments="#")
        return arr[:, :3], (arr[:, 3] if arr.shape[1] > 3 else None)
    if ext == ".pcd":
        return _read_pcd(path)
    if ext == ".ply":
        return _read_ply(path)
    raise ValueError(f"unsupported point-cloud format: {ext}")


def _read_pcd(path: Path):
    with open(path, "rb") as fh:
        header: dict[str, list[str]] = {}
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path.name}: truncated PCD header")
            parts = line.decode("ascii", errors="replace").strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break
        body = fh.read()
    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    npts = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
    mode = header["DATA"][0].lower()
    if mode == "ascii":
        arr = np.loadtxt(body.decode("ascii").splitlines(), ndmin=2)
        cols = {}
        c = 0
        for f, k in zip(fields, counts):
            cols[f] = arr[:, c]
            c += k
    elif mode == "binary":
        dt = np.dtype([(f, {"F": "f", "I": "i", "U": "u"}[t] + str(s), (k,) if k > 1 else ())
                       for f, s, t, k in zip(fields, sizes, types, counts)])
        rec = np.frombuffer(body[: npts * dt.itemsize], dtype=dt, count=npts)
        cols = {f: rec[f] for f in fields}
    else:
        try:
            import open3d as o3d  # optional dependency
        except ImportError as exc:
            raise ValueError(f"{path.name}: PCD '{mode}' needs open3d (pip install open3d)") from exc
        pc = o3d.io.read_point_cloud(str(path))
        return np.asarray(pc.points), None
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
    inten = next((cols[k] for k in ("intensity", "reflectivity", "i") if k in cols), None)
    return xyz, inten


def _read_ply(path: Path):
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path.name}: not a PLY file")
        fmt, n_vertex, props, in_vertex = None, 0, [], False
        while True:
            line = fh.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n_vertex = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path.name}: list properties on vertices are not supported")
                props.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                break
        body = fh.read()
    names = [p[0] for p in props]
    if fmt == "ascii":
        arr = np.loadtxt(body.decode("ascii").splitlines()[:n_vertex], ndmin=2)
        cols = {n: arr[:, i] for i, n in enumerate(names)}
    elif fmt == "binary_little_endian":
        tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
                "char": "i1", "int8": "i1", "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
                "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}
        dt = np.dtype([(n, tmap[t]) for n, t in props])
        rec = np.frombuffer(body[: n_vertex * dt.itemsize], dtype=dt, count=n_vertex)
        cols = {n: rec[n] for n in names}
    else:
        raise ValueError(f"{path.name}: PLY format '{fmt}' not supported")
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
    inten = next((cols[k] for k in ("intensity", "scalar_intensity", "reflectivity") if k in cols), None)
    return xyz, inten


def write_kitti_bin(path: Path, xyz: np.ndarray, intensity: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.hstack([xyz.astype(np.float32), intensity.reshape(-1, 1).astype(np.float32)]).tofile(path)


def resolve_data_path(rel: str) -> Path:
    """Resolve a user-supplied path, restricted to the data/ directory."""
    p = (DATA_ROOT / rel).resolve()
    if DATA_ROOT.resolve() not in p.parents and p != DATA_ROOT.resolve():
        raise ValueError("file_sequence must be inside the data/ directory")
    if not p.exists():
        raise ValueError(f"not found: data/{rel}")
    return p


def list_sequences() -> list[dict]:
    out = []
    if not DATA_ROOT.exists():
        return out
    for p in sorted(DATA_ROOT.rglob("*")):
        rel = p.relative_to(DATA_ROOT).as_posix()
        if p.is_dir():
            n = sum(1 for f in p.iterdir() if f.suffix.lower() in SUPPORTED)
            if n:
                out.append({"path": rel, "kind": "sequence", "frames": n})
        elif p.suffix.lower() in SUPPORTED and p.parent == DATA_ROOT / "uploads":
            out.append({"path": rel, "kind": "file", "frames": 1})
    return out


class FileSequence:
    """Frame source that replays files from disk."""

    def __init__(self, rel_path: str, loop: bool = True, period_s: float = 0.1):
        root = resolve_data_path(rel_path)
        self.files = [root] if root.is_file() else sorted(
            f for f in root.iterdir() if f.suffix.lower() in SUPPORTED)
        if not self.files:
            raise ValueError(f"no supported point-cloud files in data/{rel_path}")
        self.loop = loop
        self.period = period_s
        self.name = rel_path

    def __len__(self) -> int:
        return len(self.files)

    def frame(self, frame_index: int) -> PointFrame | None:
        if frame_index >= len(self.files) and not self.loop:
            return None
        path = self.files[frame_index % len(self.files)]
        import time

        t0 = time.perf_counter()
        xyz, inten = read_points(path)
        xyz, inten, _ = validate_points(xyz, inten)
        t = frame_index * self.period
        return PointFrame(xyz=xyz, intensity=inten, timestamp=np.full(len(xyz), t), frame_index=frame_index,
                          sensor_time=t, source=f"file:{path.name}", sim_ms=(time.perf_counter() - t0) * 1e3)
