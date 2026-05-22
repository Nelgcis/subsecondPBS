## Context

The user-observed symptom is now narrower than the earlier transfer-support discussion:

- `nuclear_correction=false`: primary dose path behaves as the baseline.
- `nuclear_correction=true`: final dose is lower because the primary path is reduced by `nucWeight`.
- No meaningful halo/broadening is visible, so the nuclear dose that should be added back is absent, too small, or lost before final dose accumulation.

Do not spend this change on `0..14` clipping. That issue can still affect final support, but it does not explain why correction-on can act like "subtract `nucWeight` only".

## Upstream Flow

`RayTraceDicom-main` correction-on behavior is concentrated in `kernel_wrapper.cu`:

```text
energyReader
  reads nuclear_weights_and_sigmas_*.txt
  verifies nSamples/nEnergies/energy/peakDepth/scaleFact match primary CIDD
  uploads nucWeightTex and nucSqSigmaTex

beam setup
  primRayDims from primary ray lattice
  nucRayDims = roundTo(spotGridDims.x/y, superpTile)
  devNucRayWeights = padded spot weights
  devNucSpotIdx maps primary ray center -> nuclear spot index

per layer
  spotDistInRays = beam.getSpotIdxToGantry().getDelta().x / beam.getRaySpacing().x
  fillIddAndSigma(...)
    res    = (1 - nucWeight) * rayWeight    * dIDD / mass
    nucRes = nucWeight       * nucRayWeight * dIDD / (mass * spotDist^2)
    nucRSigmaEff uses sqrt(sigmaSq + nucSqSigma + entrySigmaSq)
  tileRadCalc/devNucRSigmaEff
  kernelSuperposition(devNucIdd, devNucRSigmaEff, devBevNucDose, nucRayDims.x, ...)
  nucTransfDiv(devDoseBox, ..., bevNucDoseTex)
```

The important invariant is not just that `nucWeight` is read. The nuclear branch must also have valid `nucRayWeight`, valid mapped `nucIdx`, non-zero `nucRes`, finite `nucRSigmaEff`, successful nuclear superposition launches, and a non-empty `nucTransfDiv` launch box.

## Current Local Findings

From source comparison:

- Local `src/algorithms/idd_sigma.cu` implements the upstream formulas for primary reduction and nuclear dose generation.
- Local `src/core/raytracedicom_wrapper.cu` creates `nucWeightTex`/`nucSqSigmaTex` when `nuclearCorrection` is true and enforces that nuclear tables are present and axis-aligned.
- Local code has a separate halo lattice plan and explicit diagnostics for `spotDist`, `nucRayDims`, `paddedSpotWeights`, `NUC_IDD`, `NUC_RSIGMA`, and `SUPERP_OUTPUT_NUC_BEV`.
- The old change `trace-upstream-halo-parity-gap` still mixes this with final support/clipping. This change replaces it as the active investigation thread.

## Main Divergence Candidates

### 1. Halo lattice identity

Upstream nuclear rays are indexed by `beam.getSpotIdxToGantry()`, i.e. the physical PB spot grid. Local pybind inputs can represent spot positions as WEQ/ray indices (`spotPositionsAreIndices=true`) and also carry explicit physical spacing through `spotSpacingX/Z -> layerSpotDeltas`.

The key parity question:

```text
local correction-on halo grid == physical PB lattice?
or
local correction-on halo grid == dense WEQ/ray lattice?
```

If the local halo plan uses the dense WEQ lattice or wrong offsets, `spotDist`, `nucRayWeights`, and `nucTransfDiv` geometry can all be wrong even though `nucWeight` reduction works.

### 2. Nuclear IDD write eligibility

Local kernel gates nuclear writes with:

```text
nuclearEnabled &&
nucIdx >= 0 &&
nucRayWeights != null &&
nucMemStep > 0 &&
spotDist > 0
```

If `rayToNucSpotIdx` maps too few rays, maps the wrong rays, or maps centers outside the active primary ray grid, `nucRes` can be near zero while primary `res` still sees the full `nucWeight` subtraction.

The old hypothesis about `nucMemStep=0` in upstream needs careful treatment: upstream's constructor stores the passed value directly, so passing `0` would not auto-stride. Local memory is explicitly flattened as `nucPlaneN * tracerSteps`, so local code currently needs a positive `nucMemStep`. The real check is whether valid and invalid `nucIdx` semantics match the local flattened layout and whether enough primary rays map to nuclear cells.

### 3. Nuclear superposition

Even if `NUC_IDD` is non-zero, `devNucRSigmaEff` can make `tileRadCalc` classify no useful tiles or overflow/skip work. Local `performNuclearTileBasedSuperposition(...)` mirrors upstream's strict radius policy, so the first diagnostic is:

```text
NUC_IDD sum > 0
NUC_RSIGMA finite where NUC_IDD > 0
Nuclear superposition tile summary totalTiles > 0
SUPERP_OUTPUT_NUC_BEV sum > 0 and wider than primary in BEV coordinates
```

### 4. Nuclear transfer back to final dose

Upstream creates `bevNucDoseTex` from the active z slab starting at `beamFirstInside`, then uses `invertAndShift(maxSuperpR, maxSuperpR, -beamFirstInside)`.

Local code creates a 3D texture from the whole `devBevNucDose` buffer and also shifts by `-beamFirstInside`. This can still be valid if the z indexing convention is consistently full-depth, but it is a parity point to verify because a mismatch can make `nucTransfDiv` sample zeros even when `SUPERP_OUTPUT_NUC_BEV` is non-zero.

Required diagnostic:

```text
SUPERP_OUTPUT_NUC_BEV non-zero
nucTransfDiv willLaunch=1
probe doseIdx -> nuc BEV idx lands inside [0,nucDims+2R) and active z
final dose delta after nucTransfDiv is positive
```

## Verification Strategy

Use stage-wise oracles rather than final support width:

1. Compare correction-on/off final totals only as a symptom.
2. For correction-on, record per representative layer:
   - `nucWeightRange`
   - `primarySpotDelta`, `haloGridDelta`, `physicalPBDelta`
   - `spotDistInRays`
   - `nucRayDims`, padded weight nonzero count, valid `rayToNucSpotIdx` count
   - `NUC_IDD` sum/nnz/max and ratio to primary IDD
   - `NUC_RSIGMA` finite count where `NUC_IDD > 0`
   - nuclear tile summary and `SUPERP_OUTPUT_NUC_BEV`
   - `nucTransfDiv` launch box and a final-dose-before/after nuclear transfer delta
3. Only after these pass should final-dose broadening be interpreted. If pre-transfer halo is correct but final support remains clipped, open a separate transfer-support change.

## Decision

This new change is the active thread for correction-on nuclear no-halo behavior. Older tasks about `0..14` clipping are not part of this change.
