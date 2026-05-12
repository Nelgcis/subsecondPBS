#!/usr/bin/env python3
"""
Compare two raw final-dose payloads under an explicit layout contract.

Typical use:
  python3 tools/compare_final_dose_bins.py A.bin C.bin --dims 200 117 149 --layout xyz

Layout semantics:
  - xyz: flat payload came from a NumPy array with shape (Nx, Ny, Nz) in C-order.
         Flat index = ((x * Ny) + y) * Nz + z.
  - zyx: flat payload came from the wrapper-native layout with shape (Nz, Ny, Nx)
         in C-order. Flat index = ((z * Ny) + y) * Nx + x.

The script reshapes both inputs into the same canonical (x, y, z) view before
computing totals and per-axis slice sums. If you ever observe:
  - every z-slice says A > C, but
  - every x-slice says A < C
for the same canonical volume, that is a contradiction. It means the files or
the reshape/layout assumptions are inconsistent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_a", type=Path, help="First raw dose payload (.bin)")
    parser.add_argument("file_c", type=Path, help="Second raw dose payload (.bin)")
    parser.add_argument(
        "--dims",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help="Dose-grid dimensions for raw payloads",
    )
    parser.add_argument(
        "--dims-from-header",
        type=Path,
        help="Path to a header-bearing dose-grid bin whose first 3 int32 values are Nx, Ny, Nz",
    )
    parser.add_argument(
        "--layout",
        choices=("xyz", "zyx", "both"),
        default="xyz",
        help="How to interpret the raw payload before converting to canonical (x, y, z)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=8,
        help="How many leading slice indices to preview for each relation",
    )
    return parser.parse_args()


def load_dims(args: argparse.Namespace) -> Tuple[int, int, int]:
    if args.dims is not None:
        dims = tuple(int(v) for v in args.dims)
    elif args.dims_from_header is not None:
        header = np.fromfile(args.dims_from_header, dtype=np.int32, count=3)
        if header.size != 3:
            raise ValueError(f"could not read 3 int32 dims from {args.dims_from_header}")
        dims = tuple(int(v) for v in header.tolist())
    else:
        raise ValueError("either --dims or --dims-from-header is required")

    if any(v <= 0 for v in dims):
        raise ValueError(f"invalid dims: {dims}")
    return dims


def load_payload(path: Path, expected: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != expected:
        raise ValueError(f"{path} has {arr.size} float32 values, expected {expected}")
    return arr


def canonical_xyz(flat: np.ndarray, dims: Tuple[int, int, int], layout: str) -> np.ndarray:
    nx, ny, nz = dims
    if layout == "xyz":
        return flat.reshape((nx, ny, nz), order="C")
    if layout == "zyx":
        return flat.reshape((nz, ny, nx), order="C").transpose(2, 1, 0)
    raise ValueError(f"unsupported layout: {layout}")


def relation_counts(diff: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = np.flatnonzero(diff > 0.0)
    lt = np.flatnonzero(diff < 0.0)
    eq = np.flatnonzero(diff == 0.0)
    return gt, lt, eq


def preview_indices(name: str, idx: Iterable[int], limit: int) -> str:
    head = [int(v) for v in list(idx)[:limit]]
    if not head:
        return f"{name}=[]"
    suffix = "..." if len(head) == limit else ""
    return f"{name}={head}{suffix}"


def print_axis_summary(axis_name: str, slice_sums_a: np.ndarray, slice_sums_c: np.ndarray, preview: int) -> None:
    diff = slice_sums_a - slice_sums_c
    gt, lt, eq = relation_counts(diff)
    print(
        f"  {axis_name}-slices: A>C {gt.size}/{diff.size}, "
        f"A<C {lt.size}/{diff.size}, A=C {eq.size}/{diff.size}, "
        f"sum(diff)={float(diff.sum()):.6f}, min(diff)={float(diff.min()):.6f}, max(diff)={float(diff.max()):.6f}"
    )
    print("   ", preview_indices("A>C idx", gt, preview), preview_indices("A<C idx", lt, preview))


def run_one_layout(layout: str, file_a: Path, file_c: Path, dims: Tuple[int, int, int], preview: int) -> None:
    nx, ny, nz = dims
    expected = nx * ny * nz
    flat_a = load_payload(file_a, expected)
    flat_c = load_payload(file_c, expected)
    vol_a = canonical_xyz(flat_a, dims, layout)
    vol_c = canonical_xyz(flat_c, dims, layout)

    total_a = float(np.sum(vol_a, dtype=np.float64))
    total_c = float(np.sum(vol_c, dtype=np.float64))
    total_diff = total_a - total_c

    print(f"\nlayout={layout} -> canonical view shape=(Nx,Ny,Nz)=({nx},{ny},{nz})")
    print(
        f"  totals: A={total_a:.6f}, C={total_c:.6f}, diff={total_diff:.6f}, "
        f"A_nonzero={int(np.count_nonzero(vol_a))}, C_nonzero={int(np.count_nonzero(vol_c))}"
    )

    x_sums_a = np.sum(vol_a, axis=(1, 2), dtype=np.float64)
    x_sums_c = np.sum(vol_c, axis=(1, 2), dtype=np.float64)
    y_sums_a = np.sum(vol_a, axis=(0, 2), dtype=np.float64)
    y_sums_c = np.sum(vol_c, axis=(0, 2), dtype=np.float64)
    z_sums_a = np.sum(vol_a, axis=(0, 1), dtype=np.float64)
    z_sums_c = np.sum(vol_c, axis=(0, 1), dtype=np.float64)

    print_axis_summary("x", x_sums_a, x_sums_c, preview)
    print_axis_summary("y", y_sums_a, y_sums_c, preview)
    print_axis_summary("z", z_sums_a, z_sums_c, preview)

    invariant = np.array(
        [
            np.sum(x_sums_a - x_sums_c, dtype=np.float64),
            np.sum(y_sums_a - y_sums_c, dtype=np.float64),
            np.sum(z_sums_a - z_sums_c, dtype=np.float64),
            total_diff,
        ],
        dtype=np.float64,
    )
    spread = float(np.max(invariant) - np.min(invariant))
    print(f"  invariant check: slice-diff sums vs total-diff spread={spread:.6e}")
    if spread > 1e-2:
        print("  WARNING: slice-diff sums disagree beyond tolerance; re-check file sizes, dtype, or layout.")
    else:
        print("  invariant check passed: all axis reductions return the same total-difference sign.")

    impossible_claim = bool(np.all(z_sums_a > z_sums_c) and np.all(x_sums_a < x_sums_c))
    print(f"  impossible claim check (all z A>C and all x A<C): {impossible_claim}")
    if not impossible_claim:
        print("  note: if you observed that contradiction elsewhere, the files or reshape/layout assumptions were mixed.")


def main() -> int:
    args = parse_args()
    dims = load_dims(args)
    layouts = ("xyz", "zyx") if args.layout == "both" else (args.layout,)

    print(f"Comparing:\n  A={args.file_a}\n  C={args.file_c}\n  dims={dims}")
    for layout in layouts:
        run_one_layout(layout, args.file_a, args.file_c, dims, args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
