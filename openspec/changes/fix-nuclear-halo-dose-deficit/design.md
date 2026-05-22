# Design

## Background

`subsecond_old/` contains the pre-commit-14ce2d4 baseline. `subSecond/` (current) contains the post-commit code. The bug surfaces as ~17% plateau dose deficit when `nuclear_correction=true`. Per the audit log, nuclear IDD output of `fillIddAndSigma` is essentially zero, so the deficit equals exactly the `nucWeight × IDD` fraction that should have been delivered through the nuclear/halo path.

## Investigation Lessons (what NOT to repeat)

### Diagnosis #1: Per-layer `spotDelta` reads (E)

**Hypothesis**: wrapper reads `beam.layerSpotDeltas[i]` (new), should read `beam.spotDelta` (old) — three sites.

**Result**: reverted, no effect on dose.

**Root cause of false positive**: bindings populates `beam.spotDelta = layerSpotDeltas.front()`, so the numerical value of `chosenDeltaX` is identical between the two priority paths when per-layer spotDelta is uniform. Numerically a no-op for this user's plan.

### Diagnosis #2: `tileRadCalcDoseGated` (A)

**Hypothesis**: new kernel adds `dose > 0` filter that skips tiles with zero IDD.

**Result**: ruled out by logic. A tile with `dose == 0` in every voxel contributes nothing to dose whether or not it goes through `kernelSuperposition` (input 0 → output 0). The `minVal` argument about radius truncation also doesn't survive scrutiny: even when `minVal` is computed over only `dose > 0` voxels, the radius selected still bounds the gaussian's actual significant extent for those same voxels.

### Diagnosis #3: Hunk #13 — `hasExplicitLayerSpotDeltas` gating `buildPhysicalPBLatticeView`

**Hypothesis**: when `layerSpotDeltas` is populated AND `spotPositionsAreIndices=true`, the NEW code skips physical PB lattice construction and uses dense WEQ lattice instead, breaking halo normalization.

**Result**: reverted to OLD's "always build physical PB lattice when `spotPositionsAreIndices=true`" pattern. **Kept** because it aligns with subsecond_old and the documented contract in `buildPhysicalPBLatticeView`'s comments. But it did NOT close the 17% deficit because:
- `subsecond_old` and post-fix-`subSecond` now do the same thing here
- if `subsecond_old` also had 17% deficit (which the user has not had an opportunity to verify due to OOM on the original plan), this hunk was never the bug

### Diagnosis #4: Buffer-reuse memset / scratch sizing (agent)

**Hypothesis**: per-beam scratch buffers (`devBevNucDoseScratch` etc.) have per-layer memset that doesn't clear past data from prior layers, leaking stale dose.

**Result**: re-verified against the code. Per-layer kernel writes use per-layer dimensions for indexing/stride; per-layer texture binding (`cudaMemcpy3D` with `extent.width × sizeof(float)` pitch) reads using per-layer dims. Stale data exists in the upper region of the beam-scope allocation but is NEVER addressed by any consumer. Not the bug.

## Empirical Audit Findings

From `tmp/rtd4_audit.log` (smaller plan, `numLayers=28`, post-Hunk-#13-fix):

```
Layer 0:
  NUC_IDD dims = (768, 744, 2684)
  NUC_IDD sum  = 0      ← absolutely zero nuclear IDD
  primary IDD sum ≈ 113,667  ← primary path OK

Layer 14 (representative middle):
  NUC_IDD dims = (768, 744, 2684), nonzero = 13,216  (of 1.5 G cells)
  NUC_IDD sum  = 126
  primary IDD sum = 494,570
  nuclear/primary ratio = 0.025% (expected ~19%)
  SUPERP_OUTPUT_NUC_BEV sum = 125.694  ← matches NUC_IDD (no further loss)

Layer 27 (last):
  NUC_IDD sum = 1,232 vs primary 11,090,000 → 0.011%
```

The data point that breaks all prior hypotheses: **`SUPERP_OUTPUT_NUC_BEV ≈ NUC_IDD`**. Whatever NUC_IDD comes in, it's transferred through superposition unscathed. The bug is upstream of nuclear superposition, in `fillIddAndSigma`'s NUCLEAR_CORR write path.

## Current Theory: `nucIdx` Shift Mis-handles Sentinel

`src/algorithms/idd_sigma.cu:137-148`:

```cuda
int nucIdx = -1;
float nucRayWeight = 0.0f;
if (nuclearEnabled) {
    nucIdx = nucIdcs[idx2d];                  // could be -1 (no nuclear mapping)
    if (nucIdx >= 0) {
        nucRayWeight = nucRayWeights[nucIdx];
    }
    nucIdx += static_cast<int>(params.getFirstStep() * params.getNucMemStep());
    //   ↑ UNCONDITIONAL: -1 becomes -1 + firstStep×nucPlaneN
}
```

`src/algorithms/idd_sigma.cu:248-251`:

```cuda
if (nuclearEnabled && nucIdx >= 0) {
    bevNucIdd[nucIdx] = nucRes;
    bevNucRSigmaEff[nucIdx] = nucRSigmaEff;
    nucIdx += static_cast<int>(params.getNucMemStep());
}
```

If `nucIdcs[idx2d]` returns `-1` (this primary ray does not map to any nuclear spot — expected for most rays in a sparse halo lattice), the unconditional shift turns `-1` into a positive integer like `-1 + 76 × 571392 = 43,425,791`. This passes `>= 0` and writes `nucRes` to a "garbage" cell in `bevNucIdd`. All unmapped rays converge to the SAME garbage cell (`-1 + offset` is the same value for all of them), so they overwrite each other; the final value is whichever thread happened to win the race. Net contribution close to zero, with sparse non-zero spots where genuine valid `nucIdcs` happened to map.

This pattern is identical in upstream `kernel_wrapper.cu` lines 263-269 / 375-380 — but **upstream passes `nucMemoryStep = 0`** to the kernel (`kernel_wrapper.cu:944`), so the shift evaluates to `firstStep × 0 = 0` and `-1` stays `-1` and the `>= 0` gate correctly excludes it. SubSecond passes `iddParams.nucMemStep = layerHaloPlan->nucPlaneN` (a large positive number), which DOES make the shift non-zero and DOES trigger the trap.

## Proposed Fix

Make the shift conditional in `src/algorithms/idd_sigma.cu`:

```cuda
if (nuclearEnabled) {
    nucIdx = nucIdcs[idx2d];
    if (nucIdx >= 0) {
        nucRayWeight = nucRayWeights[nucIdx];
        nucIdx += static_cast<int>(params.getFirstStep() * params.getNucMemStep());
    }
    // else: keep nucIdx == -1, will not pass the >= 0 gate downstream
}
```

This preserves the upstream semantic (`-1` stays `-1`) regardless of what `nucMemStep` value subSecond passes.

Equivalent alternative: change the wrapper to pass `nucMemStep = 0` like upstream. This is structurally less invasive than touching the kernel, but it requires understanding whether subSecond's wrapper relies on `nucMemStep != 0` elsewhere. The kernel-side guard is safer.

## Risk Assessment

- The kernel-side fix is 1 line of code (move the `+=` into the `if (nucIdx >= 0)` block).
- It is functionally equivalent to upstream's behavior when `nucMemStep = 0`.
- It cannot worsen the dose (a `-1` write was either lost in garbage or stomping on other writes; making it a no-op is strictly safer).
- If the fix doesn't close the 17% deficit, the bug is in the halo lattice mapping itself (most rays mapping to `-1` even when they should map to valid nuclear spots). The next-step diagnostic would be a print of `valid_count / total_count` of `nucIdcs[idx2d]`.

## Verification

1. Apply the fix in `idd_sigma.cu`.
2. Rebuild.
3. Re-run the smaller plan with `RTD_INPUT_AUDIT=1`.
4. Inspect `NUC_IDD sum` for representative layers. Expected: jump from ~126 to a value that is ~`w/(1-w) ≈ 19%` of primary IDD sum.
5. Inspect final dose at plateau region: should approach `correction=off` baseline within a few percent.

If verification fails, the fix is preserved (it's correctness-direction) but is not the sole regression. Next-round investigation focuses on `buildHaloLatticePlan`'s computation of `rayToNucSpotIdx` (the `nucIdcs` array on device).