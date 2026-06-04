#!/usr/bin/env python3
"""
Compute y-slice 2D gamma pass rates for two raw final-dose grids.

The first positional file is the evaluation dose grid under validation. The
second positional file is the reference dose grid. Before gamma is calculated,
reference voxels are zeroed wherever evaluation is zero and reference is
non-zero.

Examples:
  python3 tools/gamma_dosegrid.py output/A.bin output/B.bin --dims 233 148 306

  python3 tools/gamma_dosegrid.py \
      output/bg879_beam1035_final_dose-rtd5_A.bin \
      output/bg879_beam1035_final_dose-tps.bin \
      --dims 233 148 306 --spacing 2 2 2 --csv results/gamma_y_slices.csv
"""

import argparse
import csv
import importlib
import math
import sys
import time
from collections import namedtuple
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


SliceResult = namedtuple(
    "SliceResult",
    [
        "y_index",
        "reference_max",
        "evaluation_max",
        "valid_count",
        "pass_count",
        "pass_rate_percent",
        "gamma_min",
        "gamma_mean",
        "gamma_max",
    ],
)

RunResult = namedtuple(
    "RunResult",
    [
        "engine",
        "elapsed_sec",
        "masked_reference_voxels",
        "overall_valid_count",
        "overall_pass_count",
        "overall_pass_rate_percent",
        "slices",
    ],
)


KNOWN_FLOAT32_DIMS = {
    10_552_104: (247, 247, 247),
}


def parse_dims(values, float_count):
    if values is not None:
        dims = tuple(int(v) for v in values)
    elif float_count in KNOWN_FLOAT32_DIMS:
        dims = KNOWN_FLOAT32_DIMS[float_count]
    else:
        raise ValueError(
            f"cannot infer dims for {float_count} float32 values; pass --dims NX NY NZ"
        )

    if len(dims) != 3 or any(v <= 0 for v in dims):
        raise ValueError(f"invalid dims: {dims}")

    expected = int(dims[0]) * int(dims[1]) * int(dims[2])
    if expected != float_count:
        raise ValueError(f"dims {dims} expect {expected} float32 values, got {float_count}")

    return dims


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_bin", type=Path, help="A.bin: dose grid under validation")
    parser.add_argument("reference_bin", type=Path, help="B.bin: reference dose grid")
    parser.add_argument(
        "--dims",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help="Dose-grid dimensions. If omitted, known output sizes are inferred.",
    )
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        metavar=("SX", "SY", "SZ"),
        default=(2.0, 2.0, 2.0),
        help="Voxel spacing in mm for x, y, z. Default: 2 2 2.",
    )
    parser.add_argument(
        "--layout",
        choices=("xyz", "zyx"),
        default="xyz",
        help=(
            "Raw payload layout. xyz means NumPy shape (Nx,Ny,Nz) C-order; "
            "zyx means wrapper-native shape (Nz,Ny,Nx) C-order. Default: xyz."
        ),
    )
    parser.add_argument(
        "--dose-percent-threshold",
        type=float,
        default=3.0,
        help="Gamma dose difference threshold in percent. Default: 3.",
    )
    parser.add_argument(
        "--distance-mm-threshold",
        type=float,
        default=3.0,
        help="Gamma distance-to-agreement threshold in mm. Default: 3.",
    )
    parser.add_argument(
        "--lower-percent-dose-cutoff",
        type=float,
        default=10.0,
        help="Reference dose cutoff as percent of normalisation. Default: 10.",
    )
    parser.add_argument(
        "--gamma-pass-threshold",
        type=float,
        default=1.0,
        help="Gamma value considered passing. Default: 1.",
    )
    parser.add_argument(
        "--max-gamma",
        type=float,
        default=2.0,
        help=(
            "Stop/search cap for gamma. Values above 1 preserve pass-rate while speeding "
            "calculation. Default: 2. Use --max-gamma inf for no cap with PyMedPhys."
        ),
    )
    parser.add_argument(
        "--interp-fraction",
        type=float,
        default=10.0,
        help="PyMedPhys interpolation fraction. Ignored by voxel fallback. Default: 10.",
    )
    parser.add_argument(
        "--local-gamma",
        action="store_true",
        help="Use local gamma instead of global gamma.",
    )
    parser.add_argument(
        "--global-normalisation",
        type=float,
        default=None,
        help="Dose normalisation value. Default: max of the masked reference grid.",
    )
    parser.add_argument(
        "--ram-available",
        type=int,
        default=2 * 1024**3,
        help="Bytes of RAM PyMedPhys may use. Default: 2 GiB.",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "pymedphys", "voxel"),
        default="auto",
        help="Gamma engine. auto prefers PyMedPhys and falls back to voxel. Default: auto.",
    )
    parser.add_argument(
        "--zero-atol",
        type=float,
        default=0.0,
        help="Treat evaluation values with abs(value) <= this as zero. Default: 0.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write all y-slice pass rates as CSV.",
    )
    parser.add_argument(
        "--write-masked-reference",
        type=Path,
        help="Optional path to write the masked reference grid as float32 raw bin.",
    )
    parser.add_argument(
        "--hide-slices",
        action="store_true",
        help="Only print the overall summary and worst slices.",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=10,
        help="Number of lowest-pass-rate slices to print in the summary. Default: 10.",
    )
    return parser.parse_args()


def load_flat(path):
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"{path} is empty or not readable as float32")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path} contains NaN or Inf values")
    return arr


def canonical_xyz(flat, dims, layout):
    nx, ny, nz = dims
    if layout == "xyz":
        return flat.reshape((nx, ny, nz), order="C")
    if layout == "zyx":
        return flat.reshape((nz, ny, nx), order="C").transpose(2, 1, 0)
    raise ValueError(f"unsupported layout: {layout}")


def flatten_from_canonical_xyz(vol, layout):
    if layout == "xyz":
        return np.asarray(vol, dtype=np.float32, order="C").ravel(order="C")
    if layout == "zyx":
        return np.asarray(vol.transpose(2, 1, 0), dtype=np.float32, order="C").ravel(order="C")
    raise ValueError(f"unsupported layout: {layout}")


def mask_reference(
    evaluation_flat,
    reference_flat,
    *,
    zero_atol,
):
    eval_zero = np.abs(evaluation_flat) <= zero_atol
    ref_nonzero = np.abs(reference_flat) > zero_atol
    mask = eval_zero & ref_nonzero
    masked_reference = reference_flat.copy()
    masked_reference[mask] = 0.0
    return masked_reference, int(np.count_nonzero(mask))


def finite_gamma_summary(
    y_index,
    reference_slice,
    evaluation_slice,
    gamma,
    pass_threshold,
):
    valid = np.isfinite(gamma)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return SliceResult(
            y_index=y_index,
            reference_max=float(np.max(reference_slice)),
            evaluation_max=float(np.max(evaluation_slice)),
            valid_count=0,
            pass_count=0,
            pass_rate_percent=float("nan"),
            gamma_min=float("nan"),
            gamma_mean=float("nan"),
            gamma_max=float("nan"),
        )

    valid_gamma = gamma[valid]
    pass_count = int(np.count_nonzero(valid_gamma <= pass_threshold))
    return SliceResult(
        y_index=y_index,
        reference_max=float(np.max(reference_slice)),
        evaluation_max=float(np.max(evaluation_slice)),
        valid_count=valid_count,
        pass_count=pass_count,
        pass_rate_percent=100.0 * pass_count / valid_count,
        gamma_min=float(np.min(valid_gamma)),
        gamma_mean=float(np.mean(valid_gamma, dtype=np.float64)),
        gamma_max=float(np.max(valid_gamma)),
    )


def gamma_with_pymedphys(
    axes,
    reference_slice,
    evaluation_slice,
    *,
    dose_percent_threshold,
    distance_mm_threshold,
    lower_percent_dose_cutoff,
    interp_fraction,
    max_gamma,
    local_gamma,
    global_normalisation,
    ram_available,
):
    pymedphys = importlib.import_module("pymedphys")
    max_gamma_arg = None if math.isinf(max_gamma) else max_gamma
    return pymedphys.gamma(
        axes,
        reference_slice,
        axes,
        evaluation_slice,
        dose_percent_threshold,
        distance_mm_threshold,
        lower_percent_dose_cutoff=lower_percent_dose_cutoff,
        interp_fraction=interp_fraction,
        max_gamma=max_gamma_arg,
        local_gamma=local_gamma,
        global_normalisation=global_normalisation,
        skip_once_passed=True,
        ram_available=ram_available,
    )


def voxel_offsets(
    spacing_x,
    spacing_z,
    distance_mm_threshold,
    max_gamma,
):
    if math.isinf(max_gamma):
        raise ValueError("voxel fallback requires finite --max-gamma")

    max_distance = max_gamma * distance_mm_threshold
    max_dx = int(math.floor(max_distance / spacing_x))
    max_dz = int(math.floor(max_distance / spacing_z))

    offsets = []
    for dx in range(-max_dx, max_dx + 1):
        for dz in range(-max_dz, max_dz + 1):
            distance_sq = (dx * spacing_x) ** 2 + (dz * spacing_z) ** 2
            if distance_sq <= max_distance**2 + 1e-12:
                distance_component_sq = distance_sq / distance_mm_threshold**2
                offsets.append((dx, dz, distance_component_sq))

    offsets.sort(key=lambda item: item[2])
    return offsets


def shifted_views(
    reference,
    evaluation,
    dx,
    dz,
):
    nx, nz = reference.shape
    if dx >= 0:
        ref_x = slice(0, nx - dx)
        eval_x = slice(dx, nx)
        out_x = ref_x
    else:
        ref_x = slice(-dx, nx)
        eval_x = slice(0, nx + dx)
        out_x = ref_x

    if dz >= 0:
        ref_z = slice(0, nz - dz)
        eval_z = slice(dz, nz)
        out_z = ref_z
    else:
        ref_z = slice(-dz, nz)
        eval_z = slice(0, nz + dz)
        out_z = ref_z

    return reference[ref_x, ref_z], evaluation[eval_x, eval_z], (out_x, out_z)


def gamma_with_voxel_fallback(
    reference_slice,
    evaluation_slice,
    *,
    spacing_x,
    spacing_z,
    dose_percent_threshold,
    distance_mm_threshold,
    lower_percent_dose_cutoff,
    max_gamma,
    local_gamma,
    global_normalisation,
    offsets,
):
    if global_normalisation <= 0:
        return np.full(reference_slice.shape, np.nan, dtype=np.float32)

    lower_cutoff = global_normalisation * lower_percent_dose_cutoff / 100.0
    valid = reference_slice >= lower_cutoff
    gamma_sq = np.full(reference_slice.shape, np.inf, dtype=np.float32)

    if local_gamma:
        dose_denominator = reference_slice * dose_percent_threshold / 100.0
        valid = valid & (dose_denominator > 0)
    else:
        dose_denominator = np.float32(global_normalisation * dose_percent_threshold / 100.0)
        if dose_denominator <= 0:
            return np.full(reference_slice.shape, np.nan, dtype=np.float32)

    for dx, dz, distance_component_sq in offsets:
        ref_view, eval_view, out_slice = shifted_views(reference_slice, evaluation_slice, dx, dz)
        valid_view = valid[out_slice]
        if not np.any(valid_view):
            continue

        if local_gamma:
            denom = dose_denominator[out_slice]
            dose_component_sq = np.square((eval_view - ref_view) / denom)
        else:
            dose_component_sq = np.square((eval_view - ref_view) / dose_denominator)

        candidate = dose_component_sq + distance_component_sq
        current = gamma_sq[out_slice]
        np.minimum(current, candidate, out=current, where=valid_view)

    gamma = np.sqrt(gamma_sq).astype(np.float32, copy=False)
    gamma[~valid] = np.nan
    gamma[np.isinf(gamma)] = max_gamma
    return gamma


def calculate_y_slice_gamma(
    evaluation,
    reference,
    *,
    spacing,
    engine,
    dose_percent_threshold,
    distance_mm_threshold,
    lower_percent_dose_cutoff,
    gamma_pass_threshold,
    max_gamma,
    interp_fraction,
    local_gamma,
    global_normalisation,
    ram_available,
):
    nx, ny, nz = reference.shape
    sx, _, sz = spacing
    axes = (np.arange(nx, dtype=np.float64) * sx, np.arange(nz, dtype=np.float64) * sz)
    normalisation = (
        float(global_normalisation)
        if global_normalisation is not None
        else float(np.max(reference))
    )

    use_pymedphys = engine in ("auto", "pymedphys")
    if use_pymedphys:
        try:
            importlib.import_module("pymedphys")
            importlib.import_module("numba")
            selected_engine = "pymedphys"
        except ModuleNotFoundError:
            if engine == "pymedphys":
                raise
            selected_engine = "voxel"
    else:
        selected_engine = "voxel"

    if selected_engine == "voxel":
        offsets = tuple(voxel_offsets(sx, sz, distance_mm_threshold, max_gamma))
    else:
        offsets = ()

    results = []
    for y_index in range(ny):
        reference_slice = np.asarray(reference[:, y_index, :], dtype=np.float32)
        evaluation_slice = np.asarray(evaluation[:, y_index, :], dtype=np.float32)

        if selected_engine == "pymedphys":
            gamma = gamma_with_pymedphys(
                axes,
                reference_slice,
                evaluation_slice,
                dose_percent_threshold=dose_percent_threshold,
                distance_mm_threshold=distance_mm_threshold,
                lower_percent_dose_cutoff=lower_percent_dose_cutoff,
                interp_fraction=interp_fraction,
                max_gamma=max_gamma,
                local_gamma=local_gamma,
                global_normalisation=normalisation,
                ram_available=ram_available,
            )
        else:
            gamma = gamma_with_voxel_fallback(
                reference_slice,
                evaluation_slice,
                spacing_x=sx,
                spacing_z=sz,
                dose_percent_threshold=dose_percent_threshold,
                distance_mm_threshold=distance_mm_threshold,
                lower_percent_dose_cutoff=lower_percent_dose_cutoff,
                max_gamma=max_gamma,
                local_gamma=local_gamma,
                global_normalisation=normalisation,
                offsets=offsets,
            )

        results.append(
            finite_gamma_summary(
                y_index,
                reference_slice,
                evaluation_slice,
                gamma,
                pass_threshold=gamma_pass_threshold,
            )
        )

    return selected_engine, results


def aggregate_result(
    engine,
    elapsed_sec,
    masked_reference_voxels,
    slices,
):
    valid = sum(item.valid_count for item in slices)
    passed = sum(item.pass_count for item in slices)
    pass_rate = float("nan") if valid == 0 else 100.0 * passed / valid
    return RunResult(
        engine=engine,
        elapsed_sec=elapsed_sec,
        masked_reference_voxels=masked_reference_voxels,
        overall_valid_count=valid,
        overall_pass_count=passed,
        overall_pass_rate_percent=pass_rate,
        slices=slices,
    )


def write_csv(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "y_index",
                "reference_max",
                "evaluation_max",
                "valid_count",
                "pass_count",
                "pass_rate_percent",
                "gamma_min",
                "gamma_mean",
                "gamma_max",
            ]
        )
        for item in result.slices:
            writer.writerow(
                [
                    item.y_index,
                    f"{item.reference_max:.9g}",
                    f"{item.evaluation_max:.9g}",
                    item.valid_count,
                    item.pass_count,
                    f"{item.pass_rate_percent:.9g}",
                    f"{item.gamma_min:.9g}",
                    f"{item.gamma_mean:.9g}",
                    f"{item.gamma_max:.9g}",
                ]
            )


def format_slice(item):
    return (
        f"y={item.y_index:3d} valid={item.valid_count:6d} "
        f"pass={item.pass_count:6d} rate={item.pass_rate_percent:8.4f}% "
        f"gamma[min/mean/max]={item.gamma_min:.4g}/{item.gamma_mean:.4g}/{item.gamma_max:.4g} "
        f"ref_max={item.reference_max:.6g} eval_max={item.evaluation_max:.6g}"
    )


def print_result(
    args,
    dims,
    result,
):
    print(f"evaluation={args.evaluation_bin}")
    print(f"reference={args.reference_bin}")
    print(f"dims=(Nx,Ny,Nz)={dims} spacing_mm={tuple(float(v) for v in args.spacing)} layout={args.layout}")
    print(
        "criteria="
        f"{args.dose_percent_threshold:g}%/{args.distance_mm_threshold:g}mm "
        f"cutoff={args.lower_percent_dose_cutoff:g}% "
        f"gamma_pass<={args.gamma_pass_threshold:g} "
        f"local_gamma={bool(args.local_gamma)} max_gamma={args.max_gamma:g}"
    )
    print(f"engine={result.engine} elapsed_sec={result.elapsed_sec:.3f}")
    if result.engine == "voxel":
        print("note=voxel fallback uses grid-point search only; install PyMedPhys for interpolated gamma.")
    print(f"masked_reference_voxels={result.masked_reference_voxels}")
    print(
        f"overall valid={result.overall_valid_count} pass={result.overall_pass_count} "
        f"rate={result.overall_pass_rate_percent:.4f}%"
    )

    worst = sorted(
        [item for item in result.slices if item.valid_count > 0],
        key=lambda item: item.pass_rate_percent,
    )[: max(args.worst, 0)]
    if worst:
        print("\nworst_y_slices:")
        for item in worst:
            print("  " + format_slice(item))

    if not args.hide_slices:
        print("\ny_slices:")
        for item in result.slices:
            print("  " + format_slice(item))

    if args.csv:
        print(f"\ncsv={args.csv}")
    if args.write_masked_reference:
        print(f"masked_reference_bin={args.write_masked_reference}")


def main():
    args = parse_args()

    if len(tuple(args.spacing)) != 3 or any(float(v) <= 0 for v in args.spacing):
        raise ValueError(f"invalid spacing: {args.spacing}")
    spacing = tuple(float(v) for v in args.spacing)

    if args.dose_percent_threshold <= 0:
        raise ValueError("--dose-percent-threshold must be positive")
    if args.distance_mm_threshold <= 0:
        raise ValueError("--distance-mm-threshold must be positive")
    if args.lower_percent_dose_cutoff < 0:
        raise ValueError("--lower-percent-dose-cutoff must be non-negative")
    if args.gamma_pass_threshold <= 0:
        raise ValueError("--gamma-pass-threshold must be positive")
    if args.max_gamma <= 0:
        raise ValueError("--max-gamma must be positive")

    evaluation_flat = load_flat(args.evaluation_bin)
    reference_flat = load_flat(args.reference_bin)
    if evaluation_flat.size != reference_flat.size:
        raise ValueError(
            f"file sizes differ: {args.evaluation_bin} has {evaluation_flat.size}, "
            f"{args.reference_bin} has {reference_flat.size} float32 values"
        )

    dims = parse_dims(args.dims, evaluation_flat.size)
    masked_reference_flat, masked_count = mask_reference(
        evaluation_flat,
        reference_flat,
        zero_atol=float(args.zero_atol),
    )

    evaluation = canonical_xyz(evaluation_flat, dims, args.layout)
    reference = canonical_xyz(masked_reference_flat, dims, args.layout)

    if args.write_masked_reference:
        masked_out = flatten_from_canonical_xyz(reference, args.layout)
        args.write_masked_reference.parent.mkdir(parents=True, exist_ok=True)
        masked_out.tofile(args.write_masked_reference)

    started = time.perf_counter()
    engine, slices = calculate_y_slice_gamma(
        evaluation,
        reference,
        spacing=spacing,
        engine=args.engine,
        dose_percent_threshold=float(args.dose_percent_threshold),
        distance_mm_threshold=float(args.distance_mm_threshold),
        lower_percent_dose_cutoff=float(args.lower_percent_dose_cutoff),
        gamma_pass_threshold=float(args.gamma_pass_threshold),
        max_gamma=float(args.max_gamma),
        interp_fraction=float(args.interp_fraction),
        local_gamma=bool(args.local_gamma),
        global_normalisation=args.global_normalisation,
        ram_available=int(args.ram_available),
    )
    elapsed = time.perf_counter() - started

    result = aggregate_result(engine, elapsed, masked_count, slices)
    if args.csv:
        write_csv(args.csv, result)
    print_result(args, dims, result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as exc:
        if exc.name in {"pymedphys", "numba"}:
            print(
                "ERROR: PyMedPhys gamma dependencies are not installed. Install them with "
                "`python3 -m pip install pymedphys numba`, or rerun with `--engine voxel`.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        raise
