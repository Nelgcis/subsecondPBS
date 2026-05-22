# Tasks

## 1. Preserve already-applied wrapper reverts (no further action)

These were applied earlier in the debugging session and align `subSecond/` with `subsecond_old/` on the halo-relevant code paths. They are kept as correctness-direction even though they did not close the 17% deficit on their own.

- [x] 1.1 `raytracedicom_wrapper.cu:~4314` — `expectedHaloSpotDistInRays` uses `layerHaloPlan->spotDelta.x` directly (no `layerSpotDeltas[i]` branch).
- [x] 1.2 `raytracedicom_wrapper.cu:~4339` — pre-check `preCheckDeltaX` uses `beamDeltaExplicit9` priority (matches OLD).
- [x] 1.3 `raytracedicom_wrapper.cu:~4411` — `chosenDeltaX` uses `beamDeltaExplicit` priority (matches OLD).
- [x] 1.4 `raytracedicom_wrapper.cu:~4594` — `spotSpacingMm` uses `beamDeltaExplicitSite2` priority (matches OLD).
- [x] 1.5 `raytracedicom_wrapper.cu:~3589` — `buildPhysicalPBLatticeView` is called whenever `spotPositionsAreIndices=true`, regardless of `hasExplicitLayerSpotDeltas` (Hunk #13 fix, matches OLD).
- [x] 1.6 Two diagnostic-only `layerSpotDeltas[i]` references at audit log lines (~4481/4483) are kept unchanged — they don't affect computation.

## 2. Apply the kernel-side `nucIdx` guard fix

The smallest possible code change that prevents the `-1 + offset` write trap.

- [x] 2.1 Open `src/algorithms/idd_sigma.cu` and locate the block around line 137-148 (the `if (nuclearEnabled)` block inside `fillIddAndSigmaKernel`).
- [x] 2.2 Move the `nucIdx += static_cast<int>(params.getFirstStep() * params.getNucMemStep());` statement INSIDE the existing `if (nucIdx >= 0)` block. Final form:

  ```cuda
  if (nuclearEnabled) {
      nucIdx = nucIdcs[idx2d];
      if (nucIdx >= 0) {
          nucRayWeight = nucRayWeights[nucIdx];
          nucIdx += static_cast<int>(params.getFirstStep() * params.getNucMemStep());
      }
  #if NUCLEAR_CORR == GAUSS_FIT
      suppressDistalSigmaDip = true;
  #endif
  }
  ```

- [x] 2.3 No other change should be needed: the existing `if (nuclearEnabled && nucIdx >= 0)` gates at lines 217 and 248-251 will now correctly reject `nucIdx == -1` (the sentinel value, preserved).
- [ ] 2.4 Rebuild: `cd build_verify && make -j`.

## 3. Verification

- [ ] 3.1 Run rtd4 with the smaller plan AND `RTD_INPUT_AUDIT=1`. Grep stdout for `[INPUT_AUDIT][WRAPPER][NUC_IDD layer=` lines.
- [ ] 3.2 Expected pre-fix vs post-fix `NUC_IDD sum` (Layer 14 reference): `126 → ~70,000–100,000` (i.e., ~`w/(1-w) × primary_sum ≈ 0.19 × 494,570 ≈ 94,000`). The order of magnitude shift is what proves the fix.
- [ ] 3.3 Compare final dose binary at plateau region: max relative diff vs rtd5_A (correction=off) should drop from ~17% to <3%.

## 4. If verification fails

- [ ] 4.1 Add a diagnostic print after the `nucIdx = nucIdcs[idx2d]` line (inside the kernel, using a single `if (idx2d == 0 && stepNo == params.getFirstStep()) printf(...)`-style guard) to log `nucIdx, idx2d` for layer 14. Or instead add a host-side audit in the wrapper: count `>=0` entries in `layerHaloPlan->rayToNucSpotIdx` and print the ratio. Expected: `>=0` ratio ~ `1 / (spotDist²)` (e.g., 5–10% for spotDist=4.3 in raySpacing=1).
- [ ] 4.2 If `>=0` ratio is very low (<<1%), the bug shifts to `buildHaloLatticePlan` — investigate the nuclear-spot mapping logic.
- [ ] 4.3 If `>=0` ratio looks correct, the bug may lie in `mass > 1e-2f` guard or other gates inside the kernel — instrument those next.

## 5. Memory budget follow-up (separate change)

- [ ] 5.1 The user's larger plan triggers OOM under both OLD and NEW code with halo enabled. This is OUT OF SCOPE for this change. File a separate proposal to:
  - Audit per-beam scratch buffer sizing
  - Consider falling back to per-layer alloc for large `bevDoseZBeam × maxNucRayDims` scenarios
  - Document the GPU VRAM requirement vs plan dimensions
