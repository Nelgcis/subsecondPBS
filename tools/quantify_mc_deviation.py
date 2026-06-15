#!/usr/bin/env python3
"""Quantify analytic-vs-MC deviation for the 137^3 datasets (920 layered / 908 mild).

Reads rtd/mc final-dose bins, finds the depth axis automatically (the axis along
which the summed dose shows a Bragg-like profile), then reports:
  - R80 / R90 distal range (in voxels) for rtd and mc, and their difference
  - peak depth and peak ratio
  - plateau-region mean ratio (rtd/mc) in the entrance half
This is the BASELINE before any resolution-oversampling experiment.
"""
import numpy as np, sys

DIM = 137

def load(path):
    a = np.fromfile(path, dtype=np.float32)
    assert a.size == DIM**3, f"{path}: {a.size} != {DIM**3}"
    return a.reshape(DIM, DIM, DIM)

def depth_profile(vol, axis):
    # average over the two non-depth axes
    other = tuple(i for i in range(3) if i != axis)
    return vol.mean(axis=other)

def distal_R(prof, frac):
    """Distal depth (voxel, interpolated) where prof falls to frac*max past the peak."""
    pk = np.argmax(prof); pkv = prof[pk]
    thr = frac * pkv
    for i in range(pk, len(prof)-1):
        if prof[i] >= thr >= prof[i+1]:
            # linear interp
            t = (prof[i]-thr)/(prof[i]-prof[i+1]+1e-30)
            return i + t
    return float(pk)

def smooth(p, k=3):
    ker = np.ones(k)/k
    return np.convolve(p, ker, mode='same')

EDGE = 5  # ignore first/last few voxels (entrance hot voxels / edge artifacts)

def robust_peak(p):
    q = p.copy(); q[:EDGE] = 0; q[-EDGE:] = 0
    return int(np.argmax(q))

def analyze(tag, rtd_path, mc_path):
    rtd = load(rtd_path); mc = load(mc_path)
    # pick depth axis = the one whose MC mean-profile has the largest peak/mean ratio (Bragg-like)
    best, baxis = -1, 0
    for ax in range(3):
        p = depth_profile(mc, ax)
        r = p.max()/(p.mean()+1e-30)
        if r > best: best, baxis = r, ax
    pr_rtd = smooth(depth_profile(rtd, baxis))
    pr_mc  = smooth(depth_profile(mc,  baxis))
    # normalize each to its own robust peak so shapes are comparable
    pk_rtd, pk_mc = robust_peak(pr_rtd), robust_peak(pr_mc)
    nr = pr_rtd / (pr_rtd[pk_rtd] + 1e-30)
    nm = pr_mc  / (pr_mc[pk_mc]  + 1e-30)
    r80_rtd, r80_mc = distal_R(nr,0.8), distal_R(nm,0.8)
    r90_rtd, r90_mc = distal_R(nr,0.9), distal_R(nm,0.9)
    plat_hi = max(EDGE+1, int(0.6*pk_mc))
    plat_ratio = nr[EDGE:plat_hi].mean()/(nm[EDGE:plat_hi].mean()+1e-30)
    print(f"\n===== {tag}  (depth axis = {baxis}) =====")
    print(f"  peak voxel (Bragg): rtd={pk_rtd:3d}   mc={pk_mc:3d}   Δ={pk_rtd-pk_mc:+d} vox")
    print(f"  R80 (vox):    rtd={r80_rtd:6.2f} mc={r80_mc:6.2f}  Δ={r80_rtd-r80_mc:+.2f} vox")
    print(f"  R90 (vox):    rtd={r90_rtd:6.2f} mc={r90_mc:6.2f}  Δ={r90_rtd-r90_mc:+.2f} vox")
    print(f"  plateau mean ratio rtd/mc (normalized, entrance): {plat_ratio:.3f}")
    # coarse normalized profiles for eyeball check (every 8th voxel)
    idxs = list(range(0, DIM, 8))
    print("  vox  : " + " ".join(f"{i:5d}" for i in idxs))
    print("  rtd  : " + " ".join(f"{nr[i]:5.2f}" for i in idxs))
    print("  mc   : " + " ".join(f"{nm[i]:5.2f}" for i in idxs))
    return baxis

base="output/"
analyze("bg920 LAYERED (stress)",
        base+"bg920_beam1088_final_dose_layer_rtd.bin",
        base+"bg920_beam1088_final_dose_layer_mc.bin")
analyze("bg908 MILD",
        base+"bg908_beam1069_final_dose_mid_rtd.bin",
        base+"bg908_beam1069_final_dose_mid_mc.bin")
