# Fix Nuclear Halo Dose Deficit (~17% Low in Plateau)

## Why

When `nuclear_correction=true` is enabled at runtime, the final dose in the plateau region comes out ~17% lower than `nuclear_correction=false`. The deficit is consistent and matches the typical `nucWeight` value in plateau (~16% nuclear interaction fraction), strongly indicating that **the entire nuclear contribution is being lost** somewhere between `fillIddAndSigma` and the final dose volume.

Empirical data (from `tmp/rtd4_audit.log`, smaller test plan, NEW code after Hunk #13 fix):

```
Layer 14:
  SUPERP_INPUT_IDD (primary):   sum = 494,570   nonzero = 8,514,119
  NUC_IDD:                       sum = 126        nonzero = 13,216
  ratio nuc/prim = 126 / 494570 = 0.025%
  expected (if nucWeight≈0.16):  ratio = w/(1-w) ≈ 19%
  actual is ~1/800 of expected → nuclear contribution effectively zero

Layer 0:  NUC_IDD sum = 0 (literally nothing written)
Layer 27: NUC_IDD sum = 1232 vs primary 11,090,000 → ~0.011%
```

The math closes: total dose = `(1 - w) · IDD_prim + 0 · IDD_nuc = 0.83 · IDD` ≈ observed 17% deficit.

This deficit was originally attributed to commit `14ce2d4` ("per-layer cudaMalloc/cudaFree 移到 beam 级复用；删掉 layer 内重复 reduce；减少无效 superposition 半径调度（IDD=0)"), but four investigation rounds + three reverts have NOT changed the deficit, suggesting the bug may pre-date this commit or live in shared code (idd_sigma.cu, untouched by the commit).

## What Changes

This change does three things:

1. **Documents the four failed diagnoses** so future debuggers don't repeat them.
2. **Consolidates the partial reverts already applied** that align subSecond with `subsecond_old` (pre-commit baseline) and upstream `RayTraceDicom-main` patterns — these are kept because they are correctness-direction even if they don't close the deficit.
3. **Proposes a focused experimental fix** in `src/algorithms/idd_sigma.cu` (the `fillIddAndSigma` kernel's nuclear branch), where the unconditional `nucIdx += firstStep × nucMemStep` shift can convert a sentinel `-1` (no nuclear-spot mapping for this primary ray) into a positive integer that incorrectly passes the `nucIdx >= 0` write gate, causing all unmapped rays to stomp on a single garbage cell.

## Impact

- **Affected code**:
  - `src/core/raytracedicom_wrapper.cu` (4 sites: already-applied revert, preserved)
  - `src/algorithms/idd_sigma.cu` (new fix candidate)
- **Affected behavior**: only the `runtimeNuclearEnabled == true` code path; `correction=off` is unchanged.
- **Risk**: low for the wrapper reverts (numerically aligned with subsecond_old). Medium for the idd_sigma.cu guard, since the same kernel pattern exists upstream — but upstream avoids the same trap by passing `nucMemoryStep = 0` to the kernel, which subSecond does not.

## Non-goals

- Performance optimization. The `commit 14ce2d4` memory optimizations (per-beam scratch reuse, deletion of redundant layer-internal reduces) are preserved.
- Fixing the separate OOM issue that affects larger plans. That is a memory-budget topic, tracked as future work.
- Re-implementing the nuclear superposition kernel or coord transforms; those have been verified identical between OLD and NEW.
- Modifying upstream `RayTraceDicom-main/`.
