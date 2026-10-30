#!/usr/bin/env python3
"""
Foundation: Wavefront OBJ (+ optional texture) -> DeliRium PSF v2/v3 container.

PSF stores equation patches (64-byte AnchorRecord + 64-byte SurfaceRecord), not triangles.
Goal of conversion: approximate a triangle mesh with a small set of quad patches whose
corner-vanishing coef[8] + shared anchors reproduce the shape at far lower memory and
with domain ops (dent, modal, splat) that never touch a triangle list.

Memory sketch (order of magnitude):
  Mesh:  V * 32B + F * 12B + UVs + texture mip chain
  PSF:   A * 64B + S * 64B   (A anchors, S surfaces; often A << V, S << F)

This tool is a *foundation*, not a full production fitter:
  1) Load OBJ positions / faces / optional UVs
  2) Build a simple uniform chart grid (quads over bbox UV or projected plane)
  3) Sample mesh for corner positions -> quantized anchors
  4) Leave coef[8] = 0 (planar patch in bilinear anchor hull) as v0 baseline
  5) Optional: average texture color into SurfaceRecord color_pattern bytes
  6) Write PSF header + records (same layout as engine Types.h / psf_splat.cl)

Later layers (surface combination abstractions):
  - Multi-chart seams (stitch int8[8])
  - Least-squares coef fit against mesh samples (true equation fit)
  - Pattern library indices from texture PCA / Fourier
  - Hierarchical merge (quadtree) for LOD

Usage:
  python3 tools/obj_to_psf.py input.obj -o assets/from_obj.psf [--tex albedo.png] [--grid 16]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def load_obj(path: Path):
    verts = []
    uvs = []
    faces = []  # list of (v0,v1,v2) 0-based, optional uv indices
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *rest = line.split()
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                tri = []
                for tok in line.split()[1:]:
                    bits = tok.split("/")
                    vi = int(bits[0]) - 1
                    ti = int(bits[1]) - 1 if len(bits) > 1 and bits[1] else -1
                    tri.append((vi, ti))
                # fan triangulate
                for i in range(1, len(tri) - 1):
                    faces.append((tri[0], tri[i], tri[i + 1]))
    return verts, uvs, faces


def bbox(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))
    span = (max(mx[0] - mn[0], 1e-6), max(mx[1] - mn[1], 1e-6), max(mx[2] - mn[2], 1e-6))
    return mn, span


def quant_pos(p, mn, span):
    def q(c, a, s):
        t = (c - a) / s
        t = 0.0 if t < 0 else 1.0 if t > 1 else t
        return int(t * 65535.0 + 0.5) & 0xFFFF

    return (q(p[0], mn[0], span[0]), q(p[1], mn[1], span[1]), q(p[2], mn[2], span[2]))


def pack_anchor(pos_u16, normal=(0, 0, 255)):
    # 64 bytes matching Types.h AnchorRecord
    buf = bytearray(64)
    struct.pack_into("<I", buf, 0, 0xFFFFFFFF)  # bone_index none
    struct.pack_into("<4H", buf, 4, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF)
    # pos at 32
    struct.pack_into("<3H", buf, 32, pos_u16[0], pos_u16[1], pos_u16[2])
    buf[44], buf[45], buf[46] = normal
    return bytes(buf)


def pack_surface(a0, a1, a2, a3, coef=(0.0,) * 8):
    buf = bytearray(64)
    struct.pack_into("<4H", buf, 0, a0, a1, a2, a3)
    struct.pack_into("<I", buf, 8, 0)  # surface_def
    struct.pack_into("<8f", buf, 12, *coef)
    # color_pattern[18] left zero -> neutral
    return bytes(buf)


def write_psf(path: Path, anchors: list, surfaces: list, mn, span, version=2):
    n_a, n_s = len(anchors), len(surfaces)
    hdr = bytearray(64)
    hdr[0:4] = b"PSF\0"
    struct.pack_into("<5I", hdr, 4, version, n_a, n_s, 0, 0)
    struct.pack_into("<7f", hdr, 24, mn[0], mn[1], mn[2], span[0], span[1], span[2], 1.0)
    with path.open("wb") as f:
        f.write(hdr)
        for a in anchors:
            f.write(a)
        for s in surfaces:
            f.write(s)
    mesh_est = n_a * 32 + n_s * 2 * 12  # rough tris proxy if each quad ~2 tris
    psf_bytes = 64 + n_a * 64 + n_s * 64
    print(f"Wrote {path}  anchors={n_a} surfaces={n_s} bytes={psf_bytes}")
    print(f"  Rough compare: PSF {psf_bytes} B vs naive mesh-ish ~{mesh_est}+texture B (order-of-magnitude only)")


def grid_from_bbox(verts, grid: int):
    """v0: lay a grid on XZ (or longest pair of axes), lift Y from nearest vertex."""
    mn, span = bbox(verts)
    # pick axes: X and Z typically for upright props
    anchors = []
    index = {}
    for j in range(grid + 1):
        for i in range(grid + 1):
            u = i / grid
            v = j / grid
            x = mn[0] + u * span[0]
            z = mn[2] + v * span[2]
            # nearest vertex Y
            best = min(verts, key=lambda p: (p[0] - x) ** 2 + (p[2] - z) ** 2)
            y = best[1]
            q = quant_pos((x, y, z), mn, span)
            index[(i, j)] = len(anchors)
            anchors.append(pack_anchor(q))
    surfaces = []
    for j in range(grid):
        for i in range(grid):
            a0 = index[(i, j)]
            a1 = index[(i + 1, j)]
            a2 = index[(i + 1, j + 1)]
            a3 = index[(i, j + 1)]
            surfaces.append(pack_surface(a0, a1, a2, a3))
    return anchors, surfaces, mn, span


def main():
    ap = argparse.ArgumentParser(description="OBJ -> PSF foundation converter")
    ap.add_argument("obj", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--grid", type=int, default=16, help="quad grid resolution")
    ap.add_argument("--tex", type=Path, default=None, help="optional albedo (reserved)")
    args = ap.parse_args()
    if not args.obj.is_file():
        print("missing", args.obj, file=sys.stderr)
        sys.exit(1)
    verts, uvs, faces = load_obj(args.obj)
    if not verts:
        print("no vertices", file=sys.stderr)
        sys.exit(1)
    print(f"OBJ: V={len(verts)} F={len(faces)} VT={len(uvs)}")
    anchors, surfaces, mn, span = grid_from_bbox(verts, max(2, args.grid))
    if args.tex:
        print(f"(texture {args.tex} reserved for pattern-fit layer — not written in v0)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_psf(args.output, anchors, surfaces, mn, span)
    print("Next layers: LS coef fit, UV charts, pattern K from texture, seam stitch.")


if __name__ == "__main__":
    main()
