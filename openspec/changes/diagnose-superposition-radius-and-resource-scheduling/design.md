## Current Mental Model

CarbonPBS and subSecond do not compute lateral spread through the same runtime
shape.

CarbonPBS final dose is direct voxel/spot evaluation:

```text
ROI voxel + spot/subspot
  -> projectedLength
  -> WEQ texture lookup
  -> profileData sigma + beamParaData(r2,rtheta,theta2)
  -> Gaussian threshold
  -> IDD lookup
  -> atomicAdd final dose
```

subSecond RTD-backed final dose is staged:

```text
spot lattice / CPB
  -> BEV ray weights
  -> WEQ/CT tracing
  -> cumulSp + density
  -> sigmaSq evolution
  -> rSigmaEff
  -> tileRadCalc
  -> superposition radius bucket
  -> BEV-to-dose transfer
```

Therefore the radius overflow warning is a subSecond/RTD-chain diagnostic. It
does not have a one-to-one CarbonPBS warning because CarbonPBS does not batch
work into `rSigmaEff`-derived superposition radius buckets.

## Radius Overflow Formula

`tileRadCalc` takes the minimum `rSigmaEff` in each `(tile,z)` and computes:

```text
rad = min(int(KS_SIGMA_CUTOFF / (sqrt(2) * minRSigmaEff) + 0.5),
          MAX_SUPERP_RADIUS + 1)
```

Overflow means:

```text
minRSigmaEff < KS_SIGMA_CUTOFF / (sqrt(2) * (MAX_SUPERP_RADIUS + 0.5))
```

With the current constants this is about `0.06527`.

`rSigmaEff` itself is:

```text
rSigmaEff = meanVoxelWidth / (sqrt(2) * (sqrt(sigmaSq) + sigmaDelta))
```

So overflow can come from:

- `sigmaSq` too large
- voxel/ray width too small
- `sigmaDelta` or units inconsistent
- geometry causing fan voxel widths to shrink
- deep energy layers accumulating more scattering
- density/radiation-length path increasing scattering
- cutoff/afterLast allowing far distal steps to participate

## Full `rSigmaEff -> tileRadCalc -> overflow` Chain

The primary radius warning is produced after these stages:

```text
wrapper layer setup
  -> FillIddAndSigmaParams(first, afterLast, peakDepth, rangeStopDepth,
                           fan corner/delta/dist, energyIdx, entry sigma)
  -> fillIddAndSigmaKernel per ray/step
  -> bevRSigmaEff[ray, step]
  -> optional padding to superposition tile multiples
  -> tileRadCalc min(bevRSigmaEff) over each 32 x 8 tile and z step
  -> radius bucket histogram
  -> overflow bucket kMaxSuperpR + 1
  -> clamp into kMaxSuperpR and warn
```

The local primary formula in `fillIddAndSigmaKernel` is:

```text
rSigmaEff =
  0.5 * (voxelWidthX(step) + voxelWidthY(step)) /
  (sqrt(2) * (sqrt(sigmaSq(step, ray)) + 0.21))
```

`tileRadCalc` then computes:

```text
minRSigmaEff = min(rSigmaEff over one superposition tile at one z)
radius       = int(KS_SIGMA_CUTOFF / (sqrt(2) * minRSigmaEff) + 0.5)
overflow     = radius > MAX_SUPERP_RADIUS
```

With `KS_SIGMA_CUTOFF=3.0` and `MAX_SUPERP_RADIUS=32`, overflow starts at:

```text
minRSigmaEff < 3.0 / (sqrt(2) * 32.5) = 0.0652714
```

Equivalently, for a representative mean voxel width:

```text
sqrt(sigmaSq) > 32.5 * meanVoxelWidth / 3.0 - 0.21
```

The variables that can drive overflow are:

| Variable | Direction that makes overflow more likely | Source in code | Interpretation |
| --- | --- | --- | --- |
| `sigmaSq` | larger | `fillIddAndSigmaKernel` transport loop | Wider primary lateral spread from air divergence plus multiple scattering. |
| `incScat`, `incincScat` | larger | radiation-length term inside `cumulSp < peakDepth` branch | More scattering accumulated through density and inverse radiation length. |
| `density` | larger in `rRl = density * rRadiationLength(...)`; also affects distal correction | traced CT/WEQ density | Dense/high-Z-like regions can increase scattering; abnormal mapping can distort it. |
| `rRadiationLengthTex` lookup | larger | density-indexed material LUT | Material/radiation-length contract issue can inflate scattering. |
| `stepLength` / `delta.z` | larger | fan geometry / tracer step setup | Longer physical step increases scattering update. |
| `peakDepth` / energy layer | larger/deeper generally increases active path | layer energy table and cutoff setup | Deeper layers can keep the beam active longer and accumulate more scattering. |
| `afterLast`, `rangeStopDepth` | later distal stop | wrapper cutoff and kernel stop condition | More z planes participate in `tileRadCalc`, including planes with wider sigma. |
| `firstInside` / `firstOutside` | wider active interval | ray tracing through CT | Longer in-patient or in-field path increases opportunities for small `rSigmaEff`. |
| `voxelWidth(step)` | smaller | `delta.x/y`, `corner.z`, `dist.x/y`, step | Smaller fan/ray voxel width lowers numerator directly. Unit or source-distance errors show up here. |
| `rayDims`, padding, tile location | tiles covering any very small ray value | wrapper padding and tile reduction | One bad ray/step in a 32 x 8 tile determines the tile radius. |
| `rayWeight` | zero/very low disables a ray | spot rasterization | Low-weight rays are set to infinity and do not drive overflow; active weighted rays can. |
| `sigmaDelta=0.21` | larger lowers `rSigmaEff` | primary branch constant | A constant broadening term; mismatched branch constants shift radius thresholds. |

Because `tileRadCalc` uses the minimum over a tile, the warning is triggered by
the worst active ray in that tile and z plane, not by an average layer behavior.
This is why per-layer attribution needs both layer-level context and selected
tile/sample context.

## First-Pass Diagnostic Hypotheses

### CT / WEQ indirect factors

- HU-to-density or HU-to-stopping-power mismatch changes `bevDensity`,
  `bevCumulSp`, and radiation-length sampling.
- WEQ depth start/step/header mismatch changes where the beam enters and how far
  the active path runs.
- Air/lung/low-density regions change density and path geometry; high-Z or
  artifact-like regions can also perturb scattering.
- Voxel spacing, fan delta, source distance, or unit conversion errors change
  `voxelWidth(step)`, `stepLength`, and `stepVol`.

### Beam / layer direct factors

- Higher energy or larger peak depth usually allows longer active paths and more
  accumulated scattering.
- Initial spot sigma, source distance, divergence coefficients, and CarbonPBS
  `beamParaData` can disagree with RTD `entrySigmaSq` / air-divergence handling.
- Ray spacing / CPB spacing / spot lattice spacing mismatches can shrink
  `voxelWidth` relative to the physical spread.
- Per-layer cutoff mapping can keep a layer active beyond the range CarbonPBS
  would evaluate with `weqDepth < longitudalCutoff`.

## CarbonPBS vs subSecond Lateral-Spread Inputs

CarbonPBS final dose evaluates each ROI voxel against each spot/subspot. The
lateral width is obtained directly from plan/profile inputs:

```text
voxel position + spot direction/subspot offset
  -> projectedLength
  -> rayweqData(projectedLength, spot x/y) => weqDepth
  -> if weqDepth < longitudalCutoff
  -> profileData(weqDepth, energyIdx) gives Gaussian weights and sigma terms
  -> beamParaData(energyIdx) gives r2/rtheta/theta2 air/geometric terms
  -> sigma2 = profileSigma^2 + subspotSigmaX^2 + subspotSigmaY^2
              + 2*rtheta*phyDepth + theta2*phyDepth^2
  -> gaussianWeight
  -> iddData(weqDepth, energyIdx)
  -> voxel dose / sparse dose coefficient
```

The relevant CarbonPBS inputs are:

| Input | Role in CarbonPBS | Radius-overflow relevance in subSecond |
| --- | --- | --- |
| `rayweqData` / `rayweqSetting` | Direct WEQ depth lookup along each spot/subspot ray. | Imported as `waterEquivalence` can replace CT tracing and set `bevCumulSp`; header start/step/dims must match ray lattice. |
| `profileData` | Mixture weights, water-depth sigma terms, and overall profile weight. | Can optionally override subSecond `bevRSigmaEff`; otherwise used mainly for audit/comparison. |
| `beamParaData` | `r2`, `rtheta`, `theta2` air/geometric broadening terms by energy/profile row. | Used by the optional Carbon-profile sigma override and by input audit. |
| `subspotData` | Per-layer subspot offset, weight, `sigmaX`, `sigmaY`. | Drives spot rasterization and entry spot sigma; mismatched units affect ray weights and `entrySigmaSq`/debug context. |
| `longitudalCutoff` | Direct distal stop: CarbonPBS skips voxel contribution when `weqDepth >= cutoff`. | subSecond maps this to layer cutoff/range-stop/afterLast behavior; mismatch can keep extra distal z steps active. |

subSecond's default RTD-backed path derives lateral spread dynamically:

```text
spot/subspot lattice + CPB/ray spacing
  -> ray weights
  -> CT tracing or imported WEQ volume
  -> bevDensity and bevCumulSp per ray/step
  -> IDD lookup and sigmaSq transport
  -> bevRSigmaEff per ray/step
  -> tileRadCalc minimum over each 32 x 8 tile and z step
  -> radius bucket scheduling
```

The relevant subSecond inputs are:

| Input | Role in subSecond | Failure mode to check |
| --- | --- | --- |
| `entrySigmaSq` | Initial/air spot variance parameter in `FillIddAndSigmaParams`. | Double counting or unit mismatch changes effective spread and can diverge from CarbonPBS profile width. |
| `sigmaSq` | Transported primary lateral variance before the `sigmaDelta` term. | Larger values reduce `rSigmaEff` and increase radius bucket cost. |
| `voxelWidth(step)` | Fan-geometry physical width of one ray cell at a step. | Smaller values reduce `rSigmaEff`; source distance, corner, delta, and unit errors are visible here. |
| `bevDensity` | Density sampled from CT or synthesized from imported WEQ. | Changes scattering and mass; abnormal CT/WEQ mapping can inflate sigma evolution. |
| `rRadiationLengthTex` | Density-to-radiation-length lookup for multiple scattering. | Material LUT or density scale mismatch directly changes `thetaSq`. |
| `raySpacing` / CPB resolution | Builds the ray lattice and BEV transfer geometry. | Too fine a ray spacing lowers voxel width and can push otherwise valid layers into radius overflow. |

The important conclusion is that CarbonPBS can have a large physical lateral
profile without paying a separate radius-bucket scheduling cost. subSecond pays
for that width through `tileRadCalc`, and one extreme ray/step in a tile can
move the whole tile into the max-radius bucket.

## Per-Layer Radius Attribution Report

The report should run on an existing plan and reuse already-computed layer
state; it should not require a new plan or synthetic beam. The first useful
format is one row per layer:

```text
layer energy energyIdx peakDepthMm cutoffMm
activeFirst activeLast activeCount
rayDims superpRayDims raySpacingMm
meanVoxelWidthFirst/Mid/Last
sigmaLimitRad32First/Mid/Last
spotSigmaX/Y profileRowIdx
rad0..rad32 overflowTiles maxBucketTiles totalTiles
overflowPct minRSigmaEff approxSigmaMax notes
```

Required collection points:

| Metric group | Collection point | Why |
| --- | --- | --- |
| Energy/layer/cutoff | wrapper before IDD/sigma and superposition | Correlates overflow with layer depth and range policy. |
| Ray and superposition dimensions | wrapper padding section | Distinguishes true geometry from padding/tile scheduling. |
| Voxel width / radius-limit sigma | `FillIddAndSigmaParams::voxelWidth` at first/mid/last active step | Identifies small numerator cases without copying full arrays. |
| Radius histogram | `performPrimaryTileBasedSuperposition` after `tileRadCalc` | Direct measurement of scheduling cost by layer. |
| Overflow samples | existing `RTD_SUPERP_OVERFLOW_DEBUG` sample path | Explains whether a few pathological rays or broad layer-wide sigma drives overflow. |
| CT/WEQ summaries | BEV tracing/import stage | Separates density/material anomalies from beam/layer geometry. |

The warning debug context implemented in this change is the minimal inline
version of this report. A later full report should aggregate the same fields
into a layer-sorted table so overflowing and non-overflowing plans can be
compared without reading thousands of warning lines.

## Existing Diagnostics And Gaps

Current switches/log surfaces that are useful for CT/WEQ diagnosis:

| Diagnostic | Existing output | Covers | Gap |
| --- | --- | --- | --- |
| `RTD_INPUT_AUDIT` pybind/wrapper entry | energy axes, CIDD rows, profile/beamPara samples, grid geometry, WEQ header/body length, LUT sizes | Input contract, units, profile row mapping, WEQ shape | Does not summarize density/radiation-length values sampled by active rays. |
| `[WEQ] Imported full WEQ volume` | WEQ dims, stored/active steps, ignored tail, first/last positive, max WEQ | WEQ header and payload sanity | Samples only the first ray for positive/max scan; not a layer/ray percentile summary. |
| `[INPUT_AUDIT][WRAPPER][WEQ_ALIGN]` | projected entry, beam first/last/passive steps, `weplMin` samples | WEQ depth alignment and active span | Needs tabular per-layer cutoff/active span correlation. |
| `DensityAndSpTracerParams` fine timing block | fan start/inc, step length, density/SP scale | CT tracing geometry and LUT scale | No HU/density/SP histogram over the active path. |
| `fillBevFromWeqVolumeKernel` | synthesizes density from WEQ depth differences | Imported-WEQ density path | No min/max/percentile density summary after synthesis. |

Current switches/log surfaces that are useful for beam/layer diagnosis:

| Diagnostic | Existing output | Covers | Gap |
| --- | --- | --- | --- |
| `[BEAM]`, `[WRAPPER_ENTRY] SPOTS` | energy count, spot counts, ray spacing, spot delta, subspot dimensions | Layer/spot input shape | Not sorted by overflow severity. |
| `[SOURCE_DIST]`, `[RTD_DEBUG_GEOM]`, `[RTD_DEBUG_FAN]` | source distance, fan corner/delta/dist | Geometry scale and voxel-width inputs | Needs explicit per-layer radius-threshold fields in one row. |
| `[SIGMA_AIR]` | peak depth, range stop, deltaZ, air-divergence coefficients, step volume | Air/divergence terms in sigma evolution | No direct comparison to CarbonPBS beamPara coefficients in same row. |
| `[CUTOFF]` | peak depth, first-pass cutoff, external cutoff, afterLast | Layer distal policy | Needs combined overflow count/histogram. |
| `[SIGMA_GRID]` / `[RTD_DEBUG_VOXEL]` | active steps, first/mid/last mean voxel width, sigma-at-radius-32 | Direct radius-limit geometry | Printed only under verbose/sigma debug; not aggregated. |
| primary overflow warning context | layer, energy, energyIdx, peak depth, cutoff, active steps, ray dims, voxel width, sigma limits, spot sigma, profile row | Immediate context for each overflow warning | Does not yet include full radius histogram or density/WEQ percentile summary. |
| `RTD_SUPERP_OVERFLOW_DEBUG` samples | overflow tile z/tile, min/max finite `rSigmaEff`, approximate sigma range, positive IDD | Worst-tile explanation | Sample-based; not a full layer report. |

For task 1.4, existing diagnostics are sufficient to identify WEQ header,
voxel/fan spacing, CT grid, LUT sizes, and active WEQ depth span. Missing
diagnostics are active-path density/radiation-length summaries and an explicit
HU/SP sanity histogram when CT tracing rather than imported WEQ is used.

For task 1.5, existing diagnostics are sufficient to identify energy, peak
depth, cutoff, spot sigma, source distance, ray spacing, CPB dimensions, spot
counts, and active-step ranges. Missing diagnostics are a compact per-layer
radius histogram and a layer-sorted table that joins these fields to
`overflowTiles`.

## Overflow Policy Decision

The first overflowing plan inspected with `RTD_SUPERP_OVERFLOW_DEBUG=1` showed
primary overflow samples with:

```text
positiveIdd=0
maxIdd=0
```

while layer-level `SIGMA_FIELD` summaries reported no overflow-like samples
among positive-IDD cells. This indicates that the observed primary radius
overflow is not a physical request from dose-carrying primary fluence. It is a
scheduling artifact caused by finite `rSigmaEff` values in far distal imported
WEQ/low-density tail regions where IDD is already zero.

The superposition kernel itself already gates useful work with:

```text
dose > 0
```

but the original radius-bucket calculation gated only on `rSigmaEff`. Therefore
zero-dose tiles could be assigned to the expensive max-radius bucket, paying
large scheduling and kernel costs without contributing dose.

Policy:

- Primary `tileRadCalc` should compute radius buckets only from cells with
  `IDD > 0`.
- Tiles with no positive IDD should be skipped and should not enter any radius
  bucket.
- Nuclear superposition remains unchanged for now because the nuclear branch
  has a stricter overflow policy and wider halo physics; changing it requires a
  separate correctness review.

This resolves the observed issue as a resource-scheduling/algorithm-limit bug,
not as expected physics and not as direct evidence of dose truncation. Imported
WEQ tail diagnostics remain useful because long zero-dose active ranges can
still waste work before superposition, but the primary max-radius warning should
no longer be triggered by zero-dose tail tiles.

## Residual Positive-IDD Overflow Diagnostics

After positive-IDD gating, remaining overflow samples may still come from far
distal planes with small but positive IDD. To distinguish true physical tail
dose from imported-WEQ or numerical tail artifacts, `RTD_SUPERP_OVERFLOW_DEBUG=1`
now enriches primary overflow samples with transport context when available:

```text
transport cumulSpRange=[min,max] densityRange=[min,max] massRange=[min,max]
  rangeStopDepth=...
positiveIddTransport iddRange=[min,max] densityRange=[min,max]
  massRange=[min,max] lowDensityPositiveIdd=...
  lowMassPositiveIdd=... belowRangeStopPositiveIdd=...
  aboveRangeStopPositiveIdd=... causeHint=...
```

The cause hint is intentionally heuristic:

- `range_stop_mismatch`: positive IDD exists beyond the layer range stop.
- `mass_amplified_tail`: positive IDD appears where mass is below the kernel's
  mass guard scale.
- `low_density_tail`: positive IDD appears in near-air WEQ/density tail.
- `positive_idd_tail`: positive IDD remains with non-negligible density/mass and
  below range stop; this needs physical/numerical review before applying an IDD
  epsilon.
- `zero_idd_tile`: diagnostic fallback for any zero-dose tile that still reaches
  the sample path.

This diagnostic is read-only and only runs for sampled overflow tiles when
`RTD_SUPERP_OVERFLOW_DEBUG=1`.

## Existing Code Observations

- `src/algorithms/complete_superposition.cu` already reports overflow count and
  can print samples when `RTD_SUPERP_OVERFLOW_DEBUG` is enabled.
- `src/core/raytracedicom_wrapper.cu` already logs `[SIGMA_GRID]` and selected
  sigma diagnostics, but not yet as a compact per-layer overflow attribution
  table.
- `src/algorithms/idd_sigma.cu` computes `sigmaSq`, `rSigmaEff`, `mass`, and
  `firstPassive`; this is the earliest place to attribute radius overflow to
  geometry, density, or energy-layer behavior.
- CarbonPBS `deviceCalDose.cu` computes lateral Gaussian width from
  `profileData` and `beamParaData` per voxel and does not pay a separate
  superposition-radius cost.

## Proposed Investigation Shape

Without creating new plans, compare existing plans and existing layers by
recording the following per layer:

- energy, energy index, peak depth, external longitudinal cutoff
- active step range: `beamFirstInside`, `afterLast`, `firstPassive`
- `weplMin/Max` or sampled `cumulSp` range
- density and radiation-length summaries along active rays
- `fanDelta`, `fanCorner`, source distances, mean `voxelWidth` first/mid/last
- `sigmaSq` or derived sigma summaries first/mid/last
- `min/percentile rSigmaEff`
- radius bucket histogram `rad0..rad32`, overflow bucket count
- positive IDD tile count and max IDD in overflow samples
- spot count, positive particle sum, spot sigma, ray spacing, CPB dimensions

The decisive table is:

```text
layer | energy | peakDepth | cutoff | activeSteps | meanWidthMin |
minRSigmaEff | approxSigmaMax | rad32Tiles | overflowTiles | notes
```

## Runtime Attribution Diagnostics

`RTD_PERF_PROFILE=1` adds a compact timing surface for comparing CarbonPBS and
subSecond on the same plan. The goal is to identify whether a slow plan is
dominated by ray-weight generation, BEV/WEQ tracing, IDD/sigma transport,
primary superposition, BEV-dose texture creation, dose transfer, or cleanup.

The beam-level summary reports:

```text
[PERF_SUMMARY] beam=... totalMs=... numLayers=... totalSpots=...
  rayDims=(x,y) tracerSteps=... maxTransferBoxVoxels=...
  slowestLayer=... slowestLayerMs=...
[PERF_SUMMARY] beam=...
  ms(setup,rayWeight,bevTrace,weplReduce,iddSigma,superp,
     bevTexture,transfer,layerCleanup,layerTotal)=(...)
```

`RTD_PERF_PROFILE_LAYERS=1` adds one row per energy layer:

```text
[PERF_LAYER] beam=... layer=i/N energy=... spots=...
  ms(alloc,iddSigma,superp,tex,transfer,cleanup,total)=(...)
  activeSteps=[first,last] activeN=...
  rayDims=(...) superpRayDims=(...)
  transferGrid=(...) transferBoxVoxels=...
```

Primary superposition also emits a radius-cost row under
`RTD_PERF_PROFILE=1`:

```text
[PERF_SUPERP] layer=... totalTiles=... skippedNoDoseTiles=...
  layerMaxSuperpR=... rad0=... rad1=... rad2=... rad4=...
  rad8=... rad16=... rad24=... rad32=...
```

Interpretation:

- High `rayWeight` suggests spot/subspot convolution or CPB-to-ray mapping is
  the bottleneck.
- High `bevTrace` / `weplReduce` suggests imported-WEQ/CT tracing and active
  range setup dominate.
- High `iddSigma` scales with `rayDims * activeSteps * numLayers`.
- High `superp` with large `rad16/rad24/rad32` indicates sigma/radius-driven
  superposition cost.
- High `transfer` with large `transferBoxVoxels` indicates dose-grid transfer
  dominates large-PTV plans.
- High `bevTexture` / `layerCleanup` indicates per-layer texture creation,
  allocation, synchronization, or cleanup debt outside the physics kernels.

## Minimal Debug Hook For The Current Warning

The current warning is emitted inside `performPrimaryTileBasedSuperposition(...)`
after `tileRadCalc` has counted radius buckets. At that point the helper knows:

- `overflowTiles`
- `threshold_rSigmaEff`
- `rayDims`
- `beamFirstInside`
- `beamFirstCalculatedPassive`
- tile radius histogram

It does not know:

- energy layer index
- physical energy
- energy table index
- peak depth
- CarbonPBS longitudinal cutoff
- spot sigma / beam parameters
- representative voxel width and sigma-at-radius-limit

Those missing fields are available at the wrapper call site immediately before:

```text
performPrimaryTileBasedSuperposition(...)
```

The least invasive design is to pass an optional debug context into the
superposition helper, for example:

```text
SuperpositionDebugContext {
  layerIdx, numLayers
  energy, energyIdx
  peakDepthMm, layerCutoffMm
  beamFirstInside, beamFirstCalculatedPassive
  rayDims, superpRayDims
  meanVoxelWidthFirst/Mid/Last
  sigmaAtRad32First/Mid/Last
  spotSigmaXmm/spotSigmaYmm
  profileRowIdx
}
```

Then the warning can become self-contained:

```text
Warning: Primary superposition required radius larger than max radius; clamping to max radius
  layer=17/42 energy=302.4MeV/u energyIdx=88 peakDepthMm=...
  activeSteps=[92,318] rayDims=(...) superpRayDims=(...)
  maxRadius=32 overflowTiles=364 threshold_rSigmaEff<0.0652714
  meanVoxelWidthMm(first,mid,last)=(...)
  sigmaAtRad32Mm(first,mid,last)=(...)
```

This is preferable to only printing a wrapper-side line before the call because
it keeps the warning and its physical context together in logs, even when many
layers are processed.

This lets us answer whether overflow correlates with:

- high energy / deep layer
- long active path
- narrow fan voxel width
- abnormal density/WEQ path
- specific spot sigma or source-distance setup
- a layer-specific cutoff or LUT reconstruction artifact

## Deferred Resource/Performance Questions

After the radius diagnosis is measurable:

- Compare subSecond vs CarbonPBS runtime by stage, not just total time.
- Separate algorithmic cost from avoidable overhead such as repeated allocation,
  texture creation, host-device synchronization, debug copies, and stream
  serialization.
- Review CUDA 12.1 optimization options only after correctness-sensitive
  diagnostics stabilize.

## Resource Scheduling Optimization Decision

The first real-plan timing profile showed the dominant avoidable cost outside
the physics kernels:

```text
ms(iddSigma, superp, transfer, layerCleanup) ~= (18, 16, 7, 79)
```

Layer cleanup alone exceeded the combined IDD/sigma and superposition kernel
time. The matching allocation rows showed another per-layer fixed cost. The
lowest-risk optimization is therefore resource lifetime shortening, not a
kernel or algorithm change.

Applied policy:

- Keep the dose algorithm and kernel order unchanged.
- Move reusable primary layer scratch from per-layer lifetime to per-beam
  lifetime:
  `devRayIdd`, `devRayRSigmaEff`, `devFirstPassive`, padded IDD/sigma buffers,
  `devBevPrimDose`, and the scalar first-passive reduction scratch.
- In nuclear builds, allocate the analogous scratch from the maximum per-beam
  halo dimensions and reuse it per layer.
- Preserve correctness by clearing reused IDD buffers to zero, reused sigma
  buffers to `inf`, reused first-passive buffers to zero, and BEV dose buffers
  to zero before each layer writes them.
- Reuse the beam-level `beamFirstInsideRT` and `beamFirstOutsideRT` values
  inside each layer. These are BEV geometry/CT tracing properties and do not
  depend on energy. Continue reducing `devFirstPassive` per layer because it is
  layer/cutoff dependent.

Deferred:

- BEV dose `cudaArray` / texture object reuse remains a second-pass
  optimization. It should be handled separately because texture object
  lifetime and bound array dimensions need a more careful contract, even though
  the same principle applies.
- Empty-layer transfer skipping should be added only after a clear zero-dose
  predicate is available outside the superposition helper.
