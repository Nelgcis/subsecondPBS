## Context

The current correction-on path shows two different symptoms that were previously mixed together:

1. `correction=on` lowers total dose relative to `correction=off`.
2. The final dose support does not broaden, and the `y=1..14` band remains zero.

The first symptom is already visible before final transfer. The latest audit log shows that the halo branch is active, but underpowered:

- `IDD layer=0 sum=113667`, `NUC_IDD layer=0 sum=1039.88`
- `IDD layer=14 sum=494570`, `NUC_IDD layer=14 sum=6921.46`
- `SUPERP_OUTPUT_NUC_BEV` is non-zero, so halo generation runs, but it is too small to compensate for the primary reduction.

The second symptom is visible in exported dose bins:

- `rtd4` total = `663833.24`
- `rtd5_A` total = `786182.58`
- `rtd4` and `rtd5_A` share the same bbox: `[88,15,150]..[144,93,206]`
- both have `y=1..14 = 0`
- TPS is wider and starts much earlier in `y`: `[78,1,140]..[154,146,216]`

The current code differs from upstream in two places that directly affect halo generation:

- In `src/core/raytracedicom_wrapper.cu:3590-3604`, the correction-on path skips `buildPhysicalPBLatticeView(...)` whenever `layerSpotDeltas` is populated. For the current pybind input, that leaves halo on the dense WEQ lattice. The log proves this with `haloGridDelta=(1,1)`.
- In `src/core/raytracedicom_wrapper.cu:4636-4638`, subSecond passes `iddParams.nucMemStep = layerHaloPlan->nucPlaneN`, while upstream passes `FillIddAndSigmaParams(..., 0, ...)` in `RayTraceDicom-main/.../kernel_wrapper.cu:941-944`. The kernel in `src/algorithms/idd_sigma.cu:139-145` still unconditionally shifts `nucIdx` by `firstStep * nucMemStep`, so current sentinel handling is not upstream-equivalent.

This change exists because the older `fix-nuclear-halo-dose-deficit` change now mixes stale hypotheses with newer evidence. The new change should preserve only the findings that still survive code and output comparison.

## Goals / Non-Goals

**Goals:**

- Restore upstream-equivalent halo lattice construction for index-encoded CarbonPBS inputs that provide physical spot spacing.
- Restore upstream-equivalent nuclear IDD indexing semantics for mapped and unmapped rays.
- Add diagnostics and regression checks that prove whether missing broadening is caused by halo-generation parity or by later transfer clipping.
- Keep the investigation reproducible against `output/...-rtd4.bin`, `output/...-rtd5_A.bin`, and upstream source.

**Non-Goals:**

- Fix the separate `0..14` transfer clipping in this change.
- Broaden or refactor unrelated superposition-radius scheduling.
- Rework TPS-side plan export contracts beyond what is needed to prove parity.
- Change upstream `RayTraceDicom-main/`.

## Decisions

### 1. Treat the halo lattice bypass as the primary parity gap

Current code:

- `src/core/raytracedicom_wrapper.cu:3590-3604` keeps `haloSpotLattice = rawSpotLattice` when `layerSpotDeltas.size() == numLayers`.
- `rawSpotLattice` for `spotPositionsAreIndices=true` is the dense WEQ/ray lattice.

Local comment contract:

- `src/core/raytracedicom_wrapper.cu:1632-1711` explicitly documents that the dense WEQ lattice is not the upstream halo contract and that fallback to the WEQ header is not laterally upstream-equivalent.

Upstream reference:

- `RayTraceDicom-main/.../kernel_wrapper.cu:901-908` maps nuclear spot indices from `beam.getSpotIdxToGantry()` rather than from a dense WEQ texel lattice.

Decision:

- The implementation shall prioritize physical PB spacing and offset for halo mapping whenever the pybind input provides explicit physical spacing. The dense WEQ lattice may remain only as an explicit fallback path with audit warnings, not as the default correction-on path for CarbonPBS imports.

Alternative considered:

- Keep the current dense WEQ halo path and only tune IDD weights. Rejected because the current log already shows `haloGridDelta=(1,1)` while the plan spacing is `4.3/3.1`, and local comments already document that this path is not upstream-equivalent.

### 2. Treat `nucMemStep` divergence as a second independent parity gap

Upstream:

- `FillIddAndSigmaParams(..., spotDistInRays, 0, beamFirstInside, afterLastStep, ...)`

Current subSecond:

- `iddParams.nucMemStep = layerHaloPlan->nucPlaneN`
- kernel code still does `nucIdx += firstStep * nucMemStep` unconditionally

Decision:

- The implementation shall make current mapped/unmapped nuclear-ray semantics match upstream. That can be done either by:
  - passing upstream-equivalent `nucMemStep=0`, or
  - keeping a non-zero step but guarding the sentinel path so `-1` remains invalid for all depth steps.

This change should evaluate both and choose the one that preserves current memory layout while matching upstream behavior.

Alternative considered:

- Ignore `nucMemStep` until after lattice fixes land. Rejected because it is a direct upstream mismatch in the same correction-on branch and can distort `NUC_IDD` even after lattice parity is restored.

### 3. Separate halo parity verification from transfer clipping verification

The output bins prove that `y=1..14` is still clipped in both `rtd4` and `rtd5_A`, so final-dose support cannot be used as the only halo-parity oracle.

Decision:

- Verification must happen in two layers:
  - pre-transfer: `NUC_IDD`, `NUC_RSIGMA`, `SUPERP_OUTPUT_NUC_BEV`, halo grid delta, spotDist, valid `rayToNucSpotIdx` occupancy
  - post-transfer: final dose total and bbox, with explicit acknowledgment that unchanged `0..14` clipping is a separate failure if pre-transfer parity looks correct

Alternative considered:

- Use final-dose broadening alone as the acceptance criterion. Rejected because it cannot distinguish missing halo from later clipping.

## Risks / Trade-offs

- `[Risk]` Physical-PB inference may still fail for some irregular plans.  
  `-> Mitigation:` keep the WEQ fallback path, but make it explicit, auditable, and non-default for the current CarbonPBS import contract.

- `[Risk]` Matching upstream `nucMemStep` behavior may expose assumptions in subSecond-specific memory layout.  
  `-> Mitigation:` add direct audits of `nucIdx`, valid mapping occupancy, and per-step indexing before relying on dose totals alone.

- `[Risk]` Fixing halo parity may increase memory or launch cost because the halo lattice becomes sparser and more physically spaced.  
  `-> Mitigation:` compare occupancy and `nucRayDims` before/after and treat any new memory issue as follow-up work, not as a reason to keep incorrect physics.

- `[Risk]` The final dose may still look clipped even after halo parity is fixed.  
  `-> Mitigation:` keep the `0..14` clipping explicitly separated and open a follow-up transfer change if needed.

## Migration Plan

1. Preserve the new change as the single source of truth for upstream halo parity.
2. Implement lattice parity first and verify `haloGridDelta`, `physicalPBDelta`, and `spotDistInRays`.
3. Implement or choose the `nucMemStep` parity fix and verify mapped/unmapped nuclear indexing.
4. Re-run the representative plan with correction on/off and compare:
   - pre-transfer halo diagnostics
   - final dose totals and bbox
5. If pre-transfer parity looks correct but `0..14` remains zero, carry that forward as a separate transfer-clipping change.

## Open Questions

- For the current CarbonPBS import path, should the wrapper always bypass dense WEQ lattice construction for correction-on halo, or only when explicit physical spacing is present and uniform enough to infer a PB lattice?
- Is the cleanest upstream match to force `nucMemStep=0`, or does subSecond need non-zero `nucMemStep` for another internal layout reason that upstream does not have?
- Once halo parity is restored, what is the expected correction-on vs correction-off total-dose relationship for this exact plan before final transfer clipping is considered?
