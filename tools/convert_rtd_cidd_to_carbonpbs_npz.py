#!/usr/bin/env python3
"""Convert RayTraceDicom proton_cumul_ddd_data.txt into a CarbonPBS-friendly LUT (.npz).

This script:
  1) Parses the RTD table format:
     [nSamples nEnergies]
     [energiesPerU (nEnergies)]
     [peakDepths   (nEnergies)]
     [scaleFacts   (nEnergies)]
     [ciddMatrix   (nEnergies*nSamples)]

  2) Builds a *global* WEPL/depth axis (constant step) up to the maximum depth covered
     by the RTD table (max over energies of (nSamples-1)/scaleFact).

  3) Resamples each energy's CIDD onto that global axis and exports both CIDD and
     segment-integral IDD (diff of CIDD) in convenient numpy arrays.

Output .npz keys:
  - enelist / energiesPerU: (nEnergies,)
  - peakDepths_mm, peakDepths_cm: (nEnergies,)
  - scaleFacts_per_mm: (nEnergies,)
  - cidd_rtd: (nEnergies, nSamples)
  - depth_axis_mm / depth_axis_cm: (nSamples,)
  - cidd_resampled_mm: (nEnergies, nSamples)
  - idd_segment_resampled: (nEnergies, nSamples)
  - idd_per_mm / idd_per_cm: (nEnergies, nSamples)
  - iddsetting_cm: (3,)  => [depthStart_cm, depthStep_cm, nSamples]

Notes:
  - RTD peakDepths are in mm for the provided reference table.
  - CarbonPBS typically uses cm as its length unit; hence the *_cm exports.
"""

import argparse
import json
import numpy as np


def parse_rtd_table(path: str):
    with open(path, 'r') as f:
        tokens = f.read().split()
    if len(tokens) < 10:
        raise RuntimeError(f"File too small: {path}")

    n_samples = int(tokens[0])
    n_energies = int(tokens[1])
    idx = 2

    energies = np.array(tokens[idx:idx + n_energies], dtype=np.float32)
    idx += n_energies
    peak_depths = np.array(tokens[idx:idx + n_energies], dtype=np.float32)
    idx += n_energies
    scale_facts = np.array(tokens[idx:idx + n_energies], dtype=np.float32)
    idx += n_energies

    expected = n_samples * n_energies
    cidd = np.array(tokens[idx:idx + expected], dtype=np.float32)
    if cidd.size != expected:
        raise RuntimeError(f"Expected {expected} CIDD values, got {cidd.size}")
    cidd = cidd.reshape((n_energies, n_samples))

    return n_samples, n_energies, energies, peak_depths, scale_facts, cidd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to proton_cumul_ddd_data.txt')
    ap.add_argument('--output', required=True, help='Output .npz path')
    ap.add_argument('--info_json', default=None, help='Optional JSON summary path')
    args = ap.parse_args()

    n_samples, n_energies, energies, peak_depths, scale_facts, cidd = parse_rtd_table(args.input)

    # Global depth axis up to max depth supported by any energy curve.
    max_depth_mm = float(np.max((n_samples - 1) / scale_facts))
    depth_axis_mm = np.linspace(0.0, max_depth_mm, n_samples, dtype=np.float32)
    depth_step_mm = float(depth_axis_mm[1] - depth_axis_mm[0])
    depth_step_cm = depth_step_mm / 10.0

    sample_idx = np.arange(n_samples, dtype=np.float32)
    cidd_resampled = np.empty((n_energies, n_samples), dtype=np.float32)
    for e in range(n_energies):
        x = np.clip(depth_axis_mm * scale_facts[e], 0.0, n_samples - 1.0)
        cidd_resampled[e] = np.interp(x, sample_idx, cidd[e])

    idd_seg = np.zeros_like(cidd_resampled)
    idd_seg[:, 1:] = cidd_resampled[:, 1:] - cidd_resampled[:, :-1]

    out = dict(
        enelist=energies,
        energiesPerU=energies,
        nEnergies=np.array([n_energies], dtype=np.int32),
        nSamples=np.array([n_samples], dtype=np.int32),
        peakDepths_mm=peak_depths,
        peakDepths_cm=peak_depths / 10.0,
        scaleFacts_per_mm=scale_facts,
        cidd_rtd=cidd,
        depth_axis_mm=depth_axis_mm,
        depth_axis_cm=depth_axis_mm / 10.0,
        cidd_resampled_mm=cidd_resampled,
        idd_segment_resampled=idd_seg,
        idd_per_mm=idd_seg / depth_step_mm,
        idd_per_cm=idd_seg / depth_step_cm,
        iddsetting_cm=np.array([0.0, depth_step_cm, float(n_samples)], dtype=np.float32),
    )

    np.savez_compressed(args.output, **out)

    info = {
        'input': args.input,
        'nSamples': n_samples,
        'nEnergies': n_energies,
        'energiesPerU_minmax': [float(energies.min()), float(energies.max())],
        'peakDepths_mm_minmax': [float(peak_depths.min()), float(peak_depths.max())],
        'scaleFacts_per_mm_minmax': [float(scale_facts.min()), float(scale_facts.max())],
        'peakDepth_scaleFact_product_mean': float(np.mean(peak_depths * scale_facts)),
        'global_depth_axis_mm': {
            'maxDepth_mm': max_depth_mm,
            'step_mm': depth_step_mm,
            'step_cm': depth_step_cm,
        },
        'output': args.output,
    }

    if args.info_json:
        with open(args.info_json, 'w') as f:
            json.dump(info, f, indent=2)

    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
