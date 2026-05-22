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

## Upstream RayTraceDicom-main vs subSecond Capability Diff

Reference upstream tree: `/home/gadolinite/CASHIM_HL/subsecond/patch10_work_step10/RayTraceDicom-main (1)/RayTraceDicom-main/`. Pure CUDA; the `owl/` subtree that lives at subSecond root is NOT used by upstream's dose pipeline, so there is no RT-core / OptiX regression to compare. The diff below is about CUDA-only patterns subSecond should have inherited.

### U1. Per-beam BEV dose buffer (upstream) vs per-layer BEV buffer (subSecond)

- Upstream allocates `devBevPrimDose` once per beam (`kernel_wrapper.cu` around line 824) and `kernelSuperposition<R>` atomicAdds into it across all layers. A single `cudaMalloc3DArray + cudaCreateTextureObject + primTransfDiv` runs once per beam outside the layer loop (`kernel_wrapper.cu` ~1126-1233).
- subSecond does `cudaMemset(devBevPrimDose, 0)` per layer (`src/core/raytracedicom_wrapper.cu:5226-5227`), then `create3DTexture` (line 5476), `primTransfDiv` (line 5674), `destroyTextureObjectAndArray` (line 5783) per layer. Same triple for nuclear (5491, 5770, 5785).
- Cost: `cudaMalloc3DArray` is one of the slowest CUDA allocators (~200µs typical). Multiplied by `numLayers × {primary, nuclear}` per beam, this is the single largest avoidable cost outside physics kernels and the most likely contributor to the `layerCleanup ~= 79 ms` observation that motivated the §"Resource Scheduling Optimization Decision" above.
- Risk to dose correctness: low. Upstream proves the per-beam accumulation pattern.

### U2. Batched per-radius superposition launches: unnecessary `cudaStreamSynchronize` injected (subSecond)

- Upstream: 33 templated `kernelSuperposition<R>` launches back-to-back on default stream 0, no syncs between launches (`kernel_wrapper.cu:1043-1075`).
- subSecond mirrors the fan-out via `LAUNCH_SUPERP_KERNEL` (`src/algorithms/complete_superposition.cu:537`), but inserts `cudaStreamSynchronize(stream)` at lines 582, 590, 791, 851, 859, 955. Only the D2H of `tileRadCtrs` (line 585) actually requires synchronization; the subsequent launches do not.
- Cost: forces host serialization at each sync; defeats kernel-queue overlap. Removing the redundant syncs reduces per-layer overhead.

### U3. Convolution path host-side allocations (subSecond regression)

- Upstream `gpu_convolution_2d.cu` is ~71 lines, two kernels (xConv / yConv), block `(32,8)`, no shared memory, no host-side allocations inside the kernel call sites.
- subSecond `src/algorithms/convolution.cu` is ~721 lines and includes per-call `cudaMalloc/cudaFree` chains (convolution.cu:387-449, 588-718) plus the family in `src/algorithms/cuda_final_dose.cu:254-280, 318, 334-335`. These run on default stream and add 5-50µs alloc latency per invocation.
- Risk: low; conversion to `cudaMallocAsync` or beam-level reuse is correctness-neutral.

### U4. Stream / sync model plumbed but unused

- Upstream declares no `cudaStream_t`; every launch hits stream 0. Single-stream simplicity.
- subSecond declares streams in `src/utils/precompiled_texture_manager.cu:87` and `src/utils/texture_pool.cu:53`, but `performPrimaryTileBasedSuperposition` is invoked with `stream=0` (`src/core/raytracedicom_wrapper.cu:5369`). The infrastructure exists; the parallelism is not exploited. This is the natural insertion point for O4 (primary || nuclear overlap).

### U5. Texture pool dead code

- `src/utils/texture_pool.cu` defines `TexturePoolManager` but nothing in the wrapper or algorithm sources acquires entries from it. Three coexisting texture utilities (`texture_pool.cu`, `precompiled_texture_manager.cu`, `texture_ultra_optimized.cu`) — only the BEV dose texture path is on the hot loop and it goes through per-layer create/destroy at `raytracedicom_wrapper.cu:5476-5498` and 5783-5785.
- Implication: the wiring exists for an O(1) per-beam BEV texture; consolidating into either `texture_pool` or a per-beam handle is straightforward.

### U6. `__launch_bounds__` absent everywhere

- `grep __launch_bounds__ src/` returns zero hits. Both `kernelSuperposition<R>` (block `(32,8)`, `src/algorithms/complete_superposition.cu:448`) and `tileRadCalc*` (block `(32,4)`, lines 309, 369) have compile-time-known block dimensions. Without `__launch_bounds__`, nvcc allocates registers conservatively, which can hurt the high-radius kernels that touch ~70 shared loads per thread.
- Upstream is no better here; this is a shared opportunity rather than a regression.

### U7. Per-layer nuclear lattice replan

- Upstream computes the nuclear plan once from `beam.getSpotIdxToGantry()` and reuses it across layers.
- subSecond rebuilds the halo/nuclear lattice per layer at `src/core/raytracedicom_wrapper.cu:3506-3513`. This duplicates host-side setup work; replaceable with a per-beam plan cache.

### U8. `__ldg` for read-only device pointers

- Neither code base uses `__ldg` on `bevDensity[idx]`, `bevCumulSp[idx]`, `nucIdcs[idx2d]`. These are read-only inside `fillIddAndSigmaKernel`. Marking them via `__ldg` (or `const __restrict__`) lets the L1.5/L2 read-only path engage.

### U9. Build-flag drift between Makefile and CMake (subSecond only)

- `CMakeLists.txt:184` sets `-use_fast_math`; `Makefile:11, 112, 120, 128` does NOT. Tests built via `make` use IEEE-compliant `expf/erff/powf/sqrtf` and will be 10-30% slower in transcendental-heavy regions than CMake builds. This shows up as misleading "subSecond is slow" reports when the harness uses the Makefile path. Restoring `-use_fast_math` to the Makefile (or unifying on CMake) is a zero-line-of-physics change. See also H9 for the accuracy trade-off.

### U10. Runtime guards vs compile-time `#ifdef`

- Upstream gates nuclear correction with `#ifdef NUCLEAR_CORR` / `#if !defined(NUCLEAR_CORR) || NUCLEAR_CORR != GAUSS_FIT` (`kernel_wrapper.cu:307-309`).
- subSecond uses runtime predicates (`nuclearEnabled`, `suppressDistalSigmaDip` in `src/algorithms/idd_sigma.cu:89-97, 132, 183`). Adds 8 loads per thread for guard evaluation; small per-thread cost but executes for every IDD/sigma sample. Net cost is sub-percent; flagging for completeness, not as a priority.

### Diff summary table

| ID | Capability | Upstream pattern | subSecond status | Priority |
| --- | --- | --- | --- | --- |
| U1 | BEV dose buffer + transfer scheduling | per-beam, single primTransfDiv | per-layer, repeated | HIGH |
| U2 | Stream-sync inside batched superp launches | none | unnecessary syncs at 6 sites | HIGH |
| U3 | Convolution host-side allocs | none | per-call malloc/free | MED |
| U4 | Multi-stream plumbing | not present | present but unused | MED |
| U5 | Texture pool wiring | one shared pool | dead-code pool + ad-hoc creates | MED |
| U6 | `__launch_bounds__` | absent (shared) | absent | MED |
| U7 | Nuclear lattice replan | per-beam | per-layer | LOW-MED |
| U8 | `__ldg` on read-only ptrs | absent (shared) | absent | LOW |
| U9 | Make/CMake fast_math parity | n/a | drifted | HIGH (build hygiene) |
| U10 | Compile-time vs runtime guards | `#ifdef` | runtime predicate | LOW |

## CUDA 12.1 Optimization Inventory For cuFinalDose

Target compute capability is **sm_86 (Ampere)** for both `Makefile:11` and `CMakeLists.txt:14-39`. CUDA 12.1 features that require sm_90 (TMA / DSMEM / Programmatic Dependent Launch) are unavailable on the current binary and are listed only for completeness. The inventory below replaces and expands the original §4 list.

| # | Optimization | Subset of subSecond it targets | Expected impact | Risk to dose correctness | sm gate |
| --- | --- | --- | --- | --- | --- |
| O1 | Per-beam BEV dose buffer + single `primTransfDiv` per beam | `src/core/raytracedicom_wrapper.cu:5226-5227, 5476-5498, 5674-5680, 5783-5785` (move out of layer loop); mirrors upstream U1 | HIGH | LOW | sm_70+ |
| O2 | `cudaMallocAsync` + `cudaMemPool` for the surviving per-layer/per-call scratch and convolution helper allocations (`src/algorithms/convolution.cu:387, 393, 396, 588, 595`; `src/algorithms/cuda_final_dose.cu:254-280, 318, 334-335`; `src/utils/utils.cu:276-284`). Also covers `devSampleXs/Ys/Steps/SigmaDebug` (`raytracedicom_wrapper.cu:4843-4846`). | HIGH on plans with many small allocs | LOW | sm_60+ (CUDA 11.2+) |
| O3 | CUDA Graph capture of the per-layer steady-state sequence: `rayWeightConv -> fillIddAndSigmaKernel -> tileRadCalcDoseGated -> batched kernelSuperposition<0..32>`. Capture once on a representative layer, replay per layer with `cudaGraphExecKernelNodeSetParams` for the by-value `FillIddAndSigmaParams` and tile counters. Apply at `raytracedicom_wrapper.cu:4593-4606, 5366-5369`. | HIGH (eliminates 30-100µs launch latency × ~33 superp kernels × N_layers) | MED (replay must update by-value params; misuse silently freezes stale params; need verifier-grade audit) | sm_70+ |
| O4 | Multi-stream parallelism: primary on stream A, nuclear on stream B, `cudaEvent` join before transfer. Drop the redundant `cudaStreamSynchronize` at `complete_superposition.cu:582, 590, 791, 851, 859, 955` — only the D2H of `tileRadCtrs` at line 585 needs the sync. | MED (nuclear is smaller than primary; <30% wall-time overlap) | LOW | sm_60+ |
| O5 | Texture / surface object pool reuse: wire the existing `src/utils/texture_pool.cu` (currently dead, see U5) into the BEV dose texture path so per-layer `cudaCreateTextureObject` is amortized. Becomes per-beam when combined with O1. | MED standalone, HIGH combined with O1 | LOW | sm_60+ |
| O6 | Warp-level `__shfl_xor_sync` reduction in `tileRadCalc` / `tileRadCalcDoseGated` (`src/algorithms/complete_superposition.cu:335-349, 401-422`). Block is `(32,4)`: x-reduction is intra-warp, shfl-friendly. Replaces shared-mem tree reduction. | LOW (kernel is small) | LOW | sm_60+ |
| O7 | `__pipeline_memcpy_async` / `cp.async` for tile load in `kernelSuperposition` (`complete_superposition.cu:456-477`); overlap global->shared with erf compute. | MED | LOW (different memory ordering — barrier discipline must be reviewed) | sm_80+ (Ampere) |
| O8 | `__launch_bounds__` on `kernelSuperposition<R>` (block `(32,8)`, `complete_superposition.cu:448`) and `tileRadCalc*` (block `(32,4)`, lines 309, 369). Setting e.g. `__launch_bounds__(256, 4)` lets nvcc pick better register counts; especially helpful for the rad-32 kernel with ~70 shared loads per thread. | MED | LOW | sm_60+ |
| O9 | Programmatic Dependent Launch (PDL) for `fillIddAndSigma -> tileRadCalcDoseGated` boundary | N/A on sm_86 | — | sm_90 only — REJECTED |
| O10 | Tensor Memory Accelerator (TMA) / DSMEM | N/A on sm_86 | — | sm_90 only — REJECTED |
| O11 | L2 cache residency hints (`cudaAccessPolicyWindow`) for the LUT textures `cumulIddTex`, `rRadiationLengthTex`, `densityTex`, `stoppingPowerTex` created once per dose call. Ampere L2 is 6 MB; LUTs are tiny and hot. | LOW-MED | LOW | sm_80+ |
| O12 | `--use_fast_math` parity for Makefile (`Makefile:11, 112, 120, 128`). CMake already enables it (`CMakeLists.txt:184`). Build-config fix only. Coordinate with H9 in the error audit (fast_math has up to 8 ULP error in `__erff` / `__expf` / `__powf`). | MED | LOW for transfer/superposition kernels; consider exempting `idd_sigma.cu` if H2/H9 prove cumulative dose drift > tolerance | n/a |
| O13 | Half-precision (`__half`) density / stopping-power 1D texture reads (`src/core/bev_ray_tracing.cu:134-135`) | LOW | HIGH — feeds IDD lookup and cumulative SP integration; REJECTED |
| O14 | Cooperative groups for the WEPL min-reduction at `src/algorithms/complete_superposition.cu:294` (`sliceMinVar<float, 128>`) | LOW | LOW | sm_60+ |
| O15 | Persistent / megakernel for the per-layer hot loop | LOW (only attractive after O3 is exhausted) | HIGH — REJECTED for now |

### Prioritization (best impact ÷ risk)

```
O1  >  O12  >  O5  >  O2  >  O3  >  O8  >  O4  >  O11  >  O7  >  O6  >  O14
```

Reject for sm_86 binaries: O9, O10, O13, O15.

### Efficiency vs accuracy: how to push cuFinalDose without losing precision

The maximum-efficiency-without-accuracy-loss path is the prioritization above stopped at O8. Beyond O8 the risk profile changes:

1. **Free lunch tier (O1, O5, O12, O8, O2)** — algebraically identical to the current pipeline. Apply unconditionally once verified by a per-plan dose diff at 1e-5 relative tolerance.
2. **Graph capture tier (O3)** — must be guarded by an integration test that compares graph-launch dose vs direct-launch dose on at least one overflowing plan and one non-overflowing plan. The risk is silent staleness of by-value params, not numerical drift; correctness is detectable only by output diff, not by hardware error.
3. **Stream overlap tier (O4)** — independent kernels operating on disjoint buffers. Correctness-neutral if and only if the primary and nuclear BEV outputs are not written to the same memory before the join event. Inspect `devBevPrimDose` vs `devBevNucDose` write sets before applying.
4. **Cache / launch tuning (O11, O7, O6, O14)** — performance-only, no semantic change.
5. **Fast-math gating** — see H9: fast_math compounds erf/exp/powf ULPs across thousands of evaluations per voxel. The recommended split is: keep `--use_fast_math` for transfer and superposition kernels; consider building `idd_sigma.cu` and `bev_ray_tracing.cu` without it (`set_source_files_properties(... PROPERTIES COMPILE_OPTIONS "-fmad=true;-prec-div=true;-prec-sqrt=true")` in CMake). This isolates the most error-sensitive kernels while letting the dose-spread kernels stay fast.

## Numerical Error Accumulation Audit

This section catalogs concrete numerical hazards in the cuFinalDose pipeline. Each hazard cites file:line and the failure mode. Hazards are listed in approximate severity for dose ripple / run-to-run reproducibility / NaN risk; the table at the end gives a compact view.

### H1. WEQ / cumulative SP integration as a left-Riemann sum

- `src/core/bev_ray_tracing.cu:135`: `cumulSp += stepLen * tex1D<float>(stoppingPowerTex, huPlus1000 * params.getSpScale() + HALF);`. Summed over ~400-1000 steps per ray.
- Failure mode: with `--use_fast_math` (FTZ on), per-step rounding error is `O(eps)` (eps ≈ 1.2e-7); total drift on `cumulSp` is `O(N·eps)` ≈ 1e-4 of value at N=1000. Harmless for IDD lookup alone, but feeds H2.
- Texture bilinear has 8 fractional bits of precision (CUDA spec). At steep HU transitions (bone/air) this stair-steps the stopping-power sample inside a single LUT bin.

### H2. CIDD differencing — catastrophic cancellation

- `src/algorithms/idd_sigma.cu:168, 203, 209, 212`: `res = rayWeight * (cumulDose - cumulDoseOld) / mass;` where both `cumulDose` and `cumulDoseOld` come from `tex2D<float>(cumulIddTex, ...)`.
- Failure mode: subtraction of two nearly-equal fp32 quantities. Near the BP peak (curve steep, step small), the difference can lose 3-5 significant digits. Combined with the texture's 8-bit fractional precision, sub-step IDD samples become noisy. This is the most likely numerical contributor to fine-grained dose ripple at peak.

### H3. Density × radiation-length × stepLength chain (squared step factor)

- `src/algorithms/idd_sigma.cu:175-176`: `rRl = density * tex1D<float>(rRadiationLengthTex, density*params.getRRlScale() + HALF);` then `thetaSq = eRefSq / (betaP*betaP) * params.getStepLength() * rRl;`
- `idd_sigma.cu:179`: `incincScat += 2.0f * thetaSq * params.getStepLength() * params.getStepLength();` — `stepLength²` factor amplifies any error in stepLength quadratically.
- Failure mode: HU noise enters `density` linearly; it then enters `rRadiationLengthTex(density·scale)` again, so the product is quadratic in density noise. The radiation-length texture's 8-bit fractional precision rebounds here.

### H4. sigmaSq evolution — negative-going post-peak variance

- `src/algorithms/idd_sigma.cu:184`: `sigmaSq -= 1.5f * (incScat + incDiv) * density;` runs only when `cumulSp >= peakDepth && !suppressDistalSigmaDip`.
- Failure mode: variance is subtracted post-peak (the empirical "dip in sigma after BP"). If `incScat + incDiv` is large or `density` jumps at material boundaries, `sigmaSq` can go negative; then `sqrtf(sigmaSq)` at line 189 returns NaN. No clamp is applied before the sqrt. Same hazard exists in upstream — both pipelines vulnerable.

### H5. Sentinel `sigmaSq = -incDiv` at loop entry

- `src/algorithms/idd_sigma.cu:131`: `float sigmaSq = -incDiv;` — explicit negative variance to compensate for the first addition of `incDiv` inside the loop.
- Failure mode: correctness depends on the very first loop iteration bringing `sigmaSq` to zero. If `firstStep != 0` or `getSigmaSqAirQuad / getSigmaSqAirLin` are mis-set, `sigmaSq` stays negative going into H4.

### H6. mass / stepVol — quadratic in step index

- `src/algorithms/idd_sigma.cu:192-195`: `mass = density * params.stepVol(stepNo)`, with `stepVol = volConst + k*volLin + k*k*volSq` (`fill_idd_and_sigma_params.cu:72`).
- Failure mode: at large step indices (k > 1000), `k² ~ 1e6` dominates the constant term; relative precision in `mass` degrades to ~1e-5 in distal steps. The `mass > 1e-2f` guard (line 197) catches divide-by-zero but not gradual drift.

### H7. Atomic accumulation into BEV dose — non-deterministic

- `src/algorithms/complete_superposition.cu:513`: `atomicAdd(outDose + outIdx, tile[...])`. Per-voxel sum of dozens to hundreds of atomics across 33 batched `kernelSuperposition<R>` launches.
- Failure mode: fp32 `atomicAdd` is non-associative; warp scheduling determines order. Run-to-run dose at any voxel can differ at the 1e-5..1e-4 level purely from atomic ordering. Cannot be fixed without rewriting to a deterministic reduction (e.g., per-bucket histograms then ordered merge).

### H8. Superposition 3σ truncation and rad-32 clamp

- `src/algorithms/complete_superposition.cu:483-490`: kernel envelope is `±(rad+0.5)·rSigmaEff` and `KS_SIGMA_CUTOFF = 3.0` (`include/core/Macro.cuh:68`).
- For rad fitting under MAX_SUPERP_RADIUS: each axis loses `1 - erf(3/sqrt(2)) ≈ 0.27%` of Gaussian; 2D residual ≈ 0.54% per evaluation.
- For rad clamped at 32 (overflow path): the tail beyond 32 voxels is silently dropped. Tile-level overflow is therefore **quiet dose loss**, not just a scheduling artifact. The current "skip zero-IDD tiles" policy partially mitigates this but does not bound the residual for positive-IDD overflow tiles.

### H9. `--use_fast_math` ULP drift on `erff` / `__expf` / `__powf` / `sqrtf`

- `CMakeLists.txt:184` enables `--use_fast_math` (FTZ on, `--prec-div=false`, `--prec-sqrt=false`, `--fmad=true`). Maps `erff` → `__erff` with up to **8 ULP error** (vs 1 ULP IEEE).
- `src/algorithms/complete_superposition.cu:481, 489`: ~1089 `erff` evaluations per tile × 64 tiles per layer × N_layers per beam. ULP errors compound quasi-statistically.
- `src/algorithms/idd_sigma.cu:173`: `__powf(peakDepth - 0.5*(cumulSp+cumulSpOld), pInv)` with small argument near peak; 8 ULP error here compounds with H2.
- Recommended mitigation: compile `idd_sigma.cu` and `bev_ray_tracing.cu` without `--use_fast_math`; keep fast_math for transfer and superposition kernels. See "Efficiency vs accuracy" sub-section above.

### H10. Hardware texture bilinear precision for physics quantities

- `src/core/bev_ray_tracing.cu:128, 134, 135`: `tex3D<float>(imVolTex,...)`, `tex1D<float>(densityTex,...)`, `tex1D<float>(stoppingPowerTex,...)` all use hardware-interpolated reads.
- Failure mode: 8-bit fractional precision in the interpolator means smooth physical quantities are sampled with stair-stepping at sub-voxel scale. At lung/bone or air/tissue boundaries (HU jumps of ±200 over one voxel), this manifests as ~1% dose noise at interfaces.

### H11. Axis-aligned voxel-boundary effects at 0°/90° gantry

- `src/core/bev_ray_tracing.cu:110`: `vec3f pos = vec3f(startPos.x + HALF, startPos.y + HALF, startPos.z + HALF);` — 0.5 offset puts rays at voxel centers.
- Failure mode: when `step.x` is exactly 1.0 (axis-aligned geometry, no rotation), each sample lands exactly on a voxel boundary; the interpolator returns the average of two neighbors, doubling the apparent density jump. Practical: dose at gantry 0°/90°/180°/270° can show slight offsets vs oblique angles.

### H12. Magic constant `sigmaDelta = 0.21f`

- `src/algorithms/idd_sigma.cu:106` (and upstream `kernel_wrapper.cu:251`): `sigmaDelta = 0.21f` is added to `sqrt(sigmaSq)` before computing `rSigmaEff` (idd_sigma.cu:189).
- Origin: undocumented in both repos ("empirical widening of beam"). Likely a ~0.21 mm fudge that compensates sub-voxel discretization smearing of the convolution kernel. Anchored to RTD-main proton commissioning.
- Direct effect on radius bucket: when `sigmaSq → 0` (entry, air), `rSigmaEff ≈ voxelWidth / (sqrt2 · 0.21)` — i.e. **0.21 sets the entry rSigmaEff floor and the entry radius bucket**. Per-model overrides at idd_sigma.cu:111-119 (the nuclear `GAUSS_FIT` branch uses 0.0 → rSigmaEff = +∞ → rad=0).
- Risk: changing 0.21 without recommissioning silently shifts radius scheduling and the entry-region dose; the value should be either documented and locked, or made a tagged input.

### H13. Magic constant `RAY_WEIGHT_CUTOFF = 1e-6`

- subSecond replaces the upstream macro with a hard-coded literal at `src/algorithms/idd_sigma.cu:78`. Rays with weight below the cutoff are set to +inf and disabled.
- Risk: if rayWeight units change (e.g. CarbonPBS uses different particle-count normalization), the gate silently zeroes valid rays; net effect is undercounted dose at low-weight layers/spots.

### H14. `__powf(peakDepth - 0.5*(cumulSp+cumulSpOld), pInv)`

- `src/algorithms/idd_sigma.cu:173`: near peak, the argument is small and `pInv ≈ 0.5650`. `__powf` under fast_math has 8 ULP error. Worst case is exactly where H2 also struggles (small denominators / small differences). Apply the same fast_math split as H9.

### Hazard summary

| ID | Category | Severity | Manifestation | Recommended next action |
| --- | --- | --- | --- | --- |
| H1 | Integration | LOW | drift ~1e-4 in cumulSp at N=1000 steps | accept; verify with longer ray paths |
| H2 | Cancellation | HIGH | dose ripple at peak | tag for diagnostic dose-diff in 3.2; consider double-precision local accumulator |
| H3 | Density chain | MED | bone/air interface noise | covered by HU/SP histogram diagnostic gap noted earlier |
| H4 | Variance subtraction | HIGH | NaN risk from negative sigmaSq | clamp `sigmaSq` to >=0 before sqrtf; verify Soukup post-peak branch is intended |
| H5 | Sentinel value | LOW (correctness depends on params) | negative sigmaSq if `firstStep != 0` | invariant check on `firstStep == 0` |
| H6 | Quadratic scaling | LOW | distal mass drift ~1e-5 | accept |
| H7 | Atomic non-associativity | HIGH (reproducibility) | run-to-run delta 1e-5..1e-4 | document; only fixable by deterministic reduction |
| H8 | Superposition truncation | MED | 0.54% per-eval residual; quiet dose loss on rad-32 overflow | bound via the IDD-gated tile policy already in place; add overflow residual estimate to diagnostics |
| H9 | fast_math ULPs | MED | drift compounds across erf/exp/powf | per-file fast_math split (mitigation above) |
| H10 | Texture bilinear | MED | ~1% noise at material interfaces | accept or move physics quantities to manual interpolation |
| H11 | Axis-aligned ray boundaries | LOW-MED | gantry 0°/90° dose offset vs oblique | jitter the 0.5 offset by a tiny amount for axis-aligned beams, or warn |
| H12 | sigmaDelta = 0.21 | HIGH (contract, not numeric) | undocumented entry-region anchor | document origin; tag as an input or commissioning constant |
| H13 | RAY_WEIGHT_CUTOFF = 1e-6 | MED (contract) | unit-sensitive silent gate | document; make tunable via input contract |
| H14 | __powf near peak | MED | compounds with H2 | covered by H9 split |

### Open question for task 3.2

To distinguish *physical dose differences from CarbonPBS* from *accumulated numerical / contract error in subSecond*, the minimum decisive diagnostic is:

1. A controlled-input plan that activates each hazard one at a time (e.g. uniform water + on-axis beam isolates H2 + H7 + H9; heterogeneous slab isolates H3 + H10; rad-32 overflow plan isolates H8).
2. A double-precision reference reduction of the same kernel sequence for a single slice (offline tool, not in the hot path) to set the floor for "what would deterministic, IEEE-strict produce".
3. A per-voxel diff of two repeated GPU runs on the same input to bound H7 reproducibility noise.

These three together let any dose-difference observation be attributed to a specific hazard rather than to "subSecond is wrong".

## Beam-Fan Box vs ROI-List Spatial Filter

subSecond and carbonPBS use **inverted** spatial-filtering strategies. Cross-engine dose comparisons are only fair when this is understood; otherwise one engine appears faster or "more correct" purely as an artifact of which voxels each touches.

### subSecond: beam-fan AABB on the dose grid

- Source of truth for the "touched region" is the BEV fan footprint, not the user-supplied ROI.
- Per layer, the 8 BEV corner points `(±MAX_SUPERP_RADIUS, ±MAX_SUPERP_RADIUS, [beamFirstInside, beamFirstCalculatedPassive])` are projected through `primRayIdxToDoseIdx` and clamped to `[0, doseDims-1]` to produce `[startIdx, maxIdx]` (`src/core/raytracedicom_wrapper.cu:5516-5560`).
- Lateral "margin" = `MAX_SUPERP_RADIUS = 32` BEV pixels (`src/core/raytracedicom_wrapper.cu:3991`; `include/core/Macro.cuh:64`). This is a superposition-scheduling extent, not a clinical safety margin.
- Axial range = `[beamFirstInside, beamFirstCalculatedPassive-1]`. Anything before `beamFirstInside` is excluded — see "Shallow Region Dose=0" below.
- Transfer kernel writes inside the box only: `src/algorithms/prim_transf_kernel.cu:11-44`. Voxels outside the box keep whatever prior beams accumulated (cross-beam accumulation relies on this).
- Accumulator is plain `*res += tmp` (non-atomic, `prim_transf_kernel.cu:39`). This is correct only because layers are serialized via `cudaDeviceSynchronize` (`raytracedicom_wrapper.cu:5682`) — confirm any future multi-stream change preserves this invariant.
- The Python `roiIdx` argument is **ignored** for spatial scoping. The only kernel that consumes `roiIndices` for a bounding box (`calculateROIRangeKernel`, `src/algorithms/cuda_final_dose.cu:66-128`) is invoked only from unit tests (`src/tests/cpb_convolution_debug.cu:198, 291`) and is dead code in the production wrapper.

### carbonPBS: ROI-list iteration with no beam-footprint pre-filter

- One thread per `(spot, roi_voxel)` pair (`patch10_mod_20260410/carbonPBS/deviceCalDose.cu:513-522`); inner loop multiplies by `nsubspot` (`:532`).
- Two scalar culls: `subspotweight < 0.001` (`:537`) and `gaussianWeight > transCutoff` after evaluation (`:569`). Depth gate `weqDepth < longitudalCutoff` at `:554`.
- Write via `atomicAdd(dose + roiIndex[gty], ...)` at `:577, :528`.
- `roiIdx` IS the spatial filter; a large dose grid is irrelevant as long as `roiIndex[]` stays small.

### Cost-curve orthogonality

```
   cost
    │                                      carbonPBS
    │                                    ╱─ O(num_roi × num_spots × nsubspot)
    │                                ╱─       (independent of beam geometry)
    │                            ╱─
    │                        ╱─
    │                    ╱─
    │     ──────────────────────────────  subSecond
    │                                      O(rayDims × layers)
    │                                      (independent of ROI size)
    └──────────────────────────────────── num_roi
```

The two engines win in different regimes:

| Scenario | subSecond | carbonPBS |
| --- | --- | --- |
| Large dose grid + small ROI tightly around PTV | mild loss (box larger than ROI) | strong win (only iterates ROI) |
| Tight ROI + large beam cross-section | **strong win** (box hugs fan) | mild loss (every spot scans full ROI) |
| Many spots / many subspots | insensitive | linear blow-up |
| ROI far larger than beam fan | **strong win** | wastes work on zero-contribution voxels |

### Architectural optimization candidates

| ID | Optimization | Locations | Expected impact | Risk |
| --- | --- | --- | --- | --- |
| O-DOSE-1 | Intersect BEV box with ROI bbox: revive `calculateROIRangeKernel` (`cuda_final_dose.cu:66-128`) for an ROI-aligned AABB, then take `box_final = box_BEV ∩ box_ROI` before launching `primTransfDiv`. | `raytracedicom_wrapper.cu:5516-5572` | MED-HIGH when ROI << BEV fan | LOW (box only shrinks) |
| O-DOSE-2 | Hoist box computation from per-layer to per-beam: lateral projection depends only on beam geometry; axial `[firstInside, firstCalculatedPassive]` is the only per-layer field. | `raytracedicom_wrapper.cu:5516-5560` | LOW-MED (per-layer overhead) | LOW |
| O-DOSE-3 | Optional strict ROI gating in `prim_transf_kernel`: skip writes to voxels outside an ROI bitmap when the user explicitly requests it. | `prim_transf_kernel.cu:11-44` | MED if Python contract requires it | MED — semantic change; coordinate with Python callers first |
| O-DOSE-4 | Reconcile box extent with H8 (3σ kernel cutoff): when `max(rSigmaEff)` × 3 exceeds `MAX_SUPERP_RADIUS = 32`, the box is **too small** rather than too large; record overflow as quiet dose loss and either widen the box or document the truncation. | `raytracedicom_wrapper.cu:5516-5560`; `complete_superposition.cu` rad-32 clamp | LOW (correctness) | LOW |
| O-DOSE-5 | Remove dead `calculateROIRangeKernel` from production or wire it in for O-DOSE-1; do not leave it as misleading code. | `src/algorithms/cuda_final_dose.cu:66-128` | n/a | n/a (hygiene) |

### Outside-the-box clamp: silent dose loss when user grid is too small

The box is clamped to `[0, doseDims-1]` at `raytracedicom_wrapper.cu:5551-5560`. If the user's dose grid is narrower than the beam fan footprint, rays whose contribution lands outside the grid are silently dropped. Dose conservation within the user grid is not enforced. This stacks with H8 (3σ kernel truncation) and rad-32 clamp (also H8) to give three independent "edge loss" mechanisms. Audit recommendation: emit a warning when `startIdx` clamps to 0 or `maxIdx` clamps to `doseDims-1` while the underlying BEV projection wanted to go past — this signals user-grid-too-small rather than legitimate corner clamping.

## Pybind Dose Deficiency vs CarbonPBS

Reported symptoms from Python callers using `tps_py/dosecal.py → cuFinalDose`:

- Total dose materially lower than carbonPBS on the same plan/CT.
- Shallow / entry-region voxels show dose = 0.
- Halo (nuclear) contribution appears absent.

These three symptoms are explained by two independent gating mechanisms (one compile-time, one runtime) plus an off-by-one in the BEV→dose box. All three resolve to known code paths.

### Nuclear-correction toggle architecture

There are **two** gates and **no** runtime back-door from Python:

```
┌──────────────────────────────────────────────────────────────────┐
│ Gate 1 (compile-time): -DNUCLEAR_CORR={SOUKUP,FLUKA,GAUSS_FIT}    │
│   CMakeLists.txt:22-36                                            │
│   If OFF (default), every halo code path is excluded by #ifdef.   │
│   No env var or Python flag can re-introduce them.                │
└──────────────────────────┬───────────────────────────────────────┘
                           │ defined → halo code compiled
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ Gate 2 (runtime): pybind argument `nuclear_correction` (bool)    │
│   src/bindings/raytracedicom_pybind.cpp:2071, 2119, 2171,         │
│                                          2219, 2266 (default False) │
│   Plumbed to `nuclearCorrection` parameter:                       │
│   raytracedicom_pybind.cpp:1626, 1864, 1947, 2040.                │
│   Wrapper assigns: runtimeNuclearEnabled = nuclearCorrection;    │
│                    raytracedicom_wrapper.cu:3483.                 │
│   Gates: halo lattice plan (3506-3520), halo BEV alloc            │
│          (~5229-5237), halo superposition (5372),                 │
│          nucTransfDiv (5688, 5764), nuclear LUT pointers          │
│          passed as nullptr when off (4596-4603).                  │
└──────────────────────────────────────────────────────────────────┘

Contract enforcement (raytracedicom_wrapper.cu:1134-1153,
                       tps_py/dosecal.py:191-225):
  • build OFF + Python True  → throw RuntimeError (contract violation)
  • build OFF + Python False → primary-only, no halo (current default)
  • build GAUSS_FIT + Python False → primary-only, halo skipped at runtime
  • build GAUSS_FIT + Python True  → primary + halo (the only "full" mode)
```

`RTD_TEST_NUCLEAR_CORRECTION` exists only in `src/tests/wrapper_integration_test.cu:3895` (`envFlagEnabled("RTD_TEST_NUCLEAR_CORRECTION", false)`) and supplies the wrapper's `nuclearCorrection` parameter for the C++ test harness. It is NOT read in pybind or the wrapper main path; setting it from the shell when calling Python has no effect.

### Side effects of nuclear toggle on PRIMARY dose

Turning halo off does not merely subtract a small halo term. It also changes three constants inside the primary IDD/sigma kernel (`src/algorithms/idd_sigma.cu:105-148`):

| Constant | `nuclear_correction = False` (or build OFF) | `nuclear_correction = True` (GAUSS_FIT) | Effect on primary dose |
| --- | --- | --- | --- |
| `eRefSq` | 198.81 (idd_sigma.cu:105) | 169.0 (idd_sigma.cu:117 via GAUSS_FIT branch) | Scattering reference squared; affects `thetaSq` |
| `sigmaDelta` | 0.21 (idd_sigma.cu:106) | 0.06 (idd_sigma.cu:118) | Direct shift on `rSigmaEff` floor; changes radius bucket scheduling |
| `suppressDistalSigmaDip` | false | true (idd_sigma.cu:145-147) | When false, post-peak branch subtracts `1.5*(incScat+incDiv)*density` from `sigmaSq` (line 184) — possible NaN risk (H4) |

Consequence: a "primary-only" build is not directly comparable to a carbonPBS "primary-only" reference. The scattering and entry-region radius differ from the halo-on build by configuration, not by physics.

### Shallow region dose = 0 — causal chain

Three independent effects stack at the entry surface, any one of which can zero shallow dose:

```
   patient surface
   ──────────────────────────────────────────────────────────────
   │  step k = 0 (in air)            density ≈ 0
   │  step k = 1 (in air)            density ≈ 0
   │  step k = firstMaterial         density > 1e-4 ← first material hit
   │  step k = firstMaterial + 1 ──────────────────────► beamFirstInside[ray]
   │                              ◄── one step off-by-one at
   │                                  raytracedicom_wrapper.cu:2043
   │  step k = firstMaterial + 2     mass ≈ density × stepVol
   │                                  if mass < 1e-2f  ──► res = 0
   │                                  (idd_sigma.cu:197)
   │                                  thin slabs / sub-voxel entry zeroed
   ──────────────────────────────────────────────────────────────
```

Mechanisms:

1. **Off-by-one in `beamFirstInside`**: `beamFirstInside[idx2d] = firstMaterial + 1` (`raytracedicom_wrapper.cu:2043`). The first material BEV step itself is the "entry step", but the per-ray flag points to the NEXT step. For rays whose `firstMaterial == beamFirstInsideRT - 1` (the earliest entry-step ray), the surface BEV slice can lose its IDD contribution.
2. **Dose-box z-clip**: `[startIdx.z, maxIdx.z]` derives from `zVals = {beamFirstInside, beamFirstCalculatedPassive-1}` (`raytracedicom_wrapper.cu:5526-5528`), then `doseIdxToPrimRayIdx` shifts by `-float(beamFirstInside)` (`:5562-5564`). Dose-grid voxels upstream of `beamFirstInside` are NEVER in the transfer launch.
3. **`mass > 1e-2f` cutoff**: `idd_sigma.cu:197` zeros any IDD slice whose `mass = density × stepVol < 1e-2`. Thin entry slivers / low-density entry voxels are dropped.

A fourth mechanism applies in imported-WEQ mode: if `weqHeader[0]` (`weqProjectedStartMm`, `raytracedicom_wrapper.cu:3778`) is set past the patient surface, BEV-tracer's `firstMaterial` slips deeper and shallow voxels are filed under "never entered". The audit print at `:3791-3801` exposes this — verify against any failing plan.

### Halo absent: standard default behavior

Halo missing is **not a bug**, it is the default behavior:

- Pybind default: `nuclear_correction = False` (`src/bindings/raytracedicom_pybind.cpp:2071` and four other entries).
- `tps_py/finalDose.py:87-90, 189` reads `dose_cal_config["nuclear_correction"]` defaulting to `False`.
- Even when build is `GAUSS_FIT`, the runtime gate at `raytracedicom_wrapper.cu:3483, 5372, 5688, 5764` skips all halo work.

A Python plan that does not explicitly set `dose_cal_config["nuclear_correction"] = True` will always produce primary-only dose, lower than a carbonPBS run that includes halo.

### Enabling Nuclear Correction From Python

There is no runtime path. Both gates must be on:

```bash
# 1. Rebuild with halo compiled in:
cmake -S . -B build -DNUCLEAR_CORR=GAUSS_FIT      # or SOUKUP / FLUKA
cmake --build build -j

# 2. In the Python plan config (dose_cal_config dict):
#    dose_cal_config["nuclear_correction"] = True

# 3. Self-check before running:
python -c "import cudaCalDoseRTD; print(cudaCalDoseRTD.rtdSupportMatrix())"
#    Expect: {'nuclear_corr_compiled': True,
#             'nuclear_corr_mode': 'GAUSS_FIT' (or SOUKUP/FLUKA), ...}
```

`RTD_TEST_NUCLEAR_CORRECTION=1` is **not** an alternative for pybind. It only affects the test harness `bin/wrapper_integration_test` (`src/tests/wrapper_integration_test.cu:3895`). Setting it before `python ...` has no effect — pybind hard-codes the flag from the Python argument and ignores environment.

Pybind input diagnostics are runtime environment switches, not compile macros:

```python
import os

# Pybind boundary only: dtype, shape, strides, converted vector summaries,
# WEQ header/body checks, and cross-count checks before subsecondWrapper.
os.environ["RTD_PYBIND_AUDIT"] = "1"

# Full input audit: enables pybind audit plus wrapper-side RTD input audit.
os.environ["RTD_INPUT_AUDIT"] = "1"

# Overflow sample diagnostics in the superposition path.
os.environ["RTD_SUPERP_OVERFLOW_DEBUG"] = "1"

# Runtime stage timing.
os.environ["RTD_PERF_PROFILE"] = "1"
```

These must be set before importing/running the extension in the process that
calls pybind. They do not change compiled physics gates; they only change
logging/diagnostics.

### Physical PB Spacing Contract For Halo

The nuclear halo path requires physical pencil-beam lattice spacing, not the
WEQ/CPB ray spacing. `tps_py/dosecal.py` already carries this as per-spot arrays:

```text
scan_points[energy_idx]["spot_spacing_x"]
scan_points[energy_idx]["spot_spacing_z"]
```

These values are generated together with `scan_points[energy_idx]["x"]` and
`["z"]`, so they share the same physical unit (mm) and spot order. The pybind
final-dose call passes them as tail keyword arguments:

```python
cuFinalDose(...,
    nuclear_correction=nuclear_correction,
    spotSpacingX=spot_spacing_x_all,
    spotSpacingZ=spot_spacing_z_all)
```

`spot_spacing_x_all` and `spot_spacing_z_all` are flattened in the same loop as
`x_all`, `z_all`, and `ene_all`, so their length must equal `nBeam` and their
row order matches `idbeamxy`, `all_energies`, and `nPar`.

The May 18 pybind audit proves that the Python arrays cross the boundary:

```text
spotSpacingX converted len=13787 ... min=3.1 max=4.3
spotSpacingZ converted len=13787 ... min=3.1 max=4.3
explicit physical PB spacing spotDelta=(3.6,3.6,0)
```

This is still not a valid halo contract. The current pybind path compresses the
per-spot/per-layer spacing to one median `beamSettings.spotDelta`. That loses the
layer-specific values (`4.3 mm` in early layers, `3.1 mm` in later layers), and it
does not update the halo lattice source. Runtime still reports:

```text
[9.24] WARN: halo/primary lattice ratio=(0.1,0.0999985)
```

That warning is decisive: `spotDistInRays` can see `beam.spotDelta`, but
`layerHaloPlan->spotDelta` is still being produced by
`buildPhysicalPBLatticeView()` (`src/core/raytracedicom_wrapper.cu:1546`) through
decoded spot-cloud inference/fallback, and can fall back to WEQ/depth-step-like
`0.1`. This is a boundary/disconnect bug, not a missing constant.

Required contract:

- Python spacing is the authoritative physical PB spacing source.
- Pybind must preserve it at layer granularity, derived from `layerInfo`, instead
  of collapsing it to a global median.
- Each layer must be internally uniform within a tight tolerance. If a layer has
  mixed finite positive spacing values, fail with a diagnostic that prints layer
  index, energy, spot count, min/max, and sample values. Do not infer a replacement.
- When explicit spacing exists, halo lattice construction must not infer spacing
  from decoded centers and must not fall back to the WEQ header.
- `layerHaloPlan->spotDelta`, `spotDistInRays`, and `nucIdxToFan.delta` must all
  use the same `layerSpotDelta[layer]`.
- Weight placement may use decoded spot centers for the layer. This uses measured
  spot positions; it does not infer lattice spacing.

The expected invariant for the failing plan is:

```text
layer 0  : layerSpotDelta ~= (4.3, 4.3) mm
layer 27 : layerSpotDelta ~= (3.1, 3.1) mm
halo/primary lattice ratio ~= 1.0 when comparing the same explicit source
spotDistInRays = layerSpotDelta.x / raySpacing.x
```

Risk boundary: this change must be fail-closed. If explicit spacing is absent,
legacy inference may remain for non-pybind callers. If explicit spacing is
present and inconsistent, the run must stop rather than silently selecting a
median, WEQ step, or point-cloud-derived spacing.

### Dose Support Clipping Against CarbonPBS Output

The observed final-dose difference is much larger than a nuclear halo weight
effect. Comparing the two `/output` files with dose shape `(233,148,306)`:

```text
RTD: sum=651220.8125 max=8.534245 nnz=214213
TPS: sum=969095.8125 max=10.811095 nnz=572302
RTD/TPS sum ratio=0.671988

RTD bbox: [88,15,150] .. [144,93,206]
TPS bbox: [78, 1,140] .. [154,146,216]
```

Per-y support confirms the user-visible failure:

```text
y=1..14: RTD sum=0 while TPS is about 8964 per slice
y=15   : RTD starts nonzero but remains lower than TPS
y=94   : RTD returns to 0 while TPS remains nonzero
y=120  : RTD is 0 while TPS is still positive
```

This points to primary dose transfer/support clipping before nuclear correction
can matter. The suspected chain is the one already documented above:

- `beamFirstInside[idx2d] = firstMaterial + 1`
  (`src/core/raytracedicom_wrapper.cu:2043`) moves the first valid material step
  one step deeper.
- Primary transfer derives `zVals` from `{beamFirstInside,
  beamFirstCalculatedPassive - 1}` (`raytracedicom_wrapper.cu:5526-5528`) and
  shifts `doseIdxToPrimRayIdx` by `-beamFirstInside`
  (`raytracedicom_wrapper.cu:5562-5564`).
- Voxels outside that projected transfer box are never launched, so they remain
  zero even when the CarbonPBS ROI-list path would still accumulate dose.

Correction strategy:

1. Separate "ray first inside material" from "transfer launch lower bound".
   `beamFirstInside` may remain a tracer diagnostic, but transfer must not use it
   as a hard lower bound that excludes entry-near or laterally broad dose support.
2. Build the primary transfer dose box from a conservative projected dose-support
   range: at minimum include the entry-near range around the first material step
   and the lateral superposition margin. The BEV texture values and existing zero
   checks should decide actual dose, not an over-tight launch box.
3. Treat `firstMaterial + 1` as suspicious independently. The lowest-risk first
   probe is to audit whether changing the transfer lower bound alone restores
   y=1..14 support; only then consider changing the tracer's first-inside value.
4. Do not change `mass > 1e-2f`, scattering constants, nuclear weights, or IDD
   tables in the same patch. Those are separate physics/numerical decisions and
   would make attribution ambiguous.
5. Add verifier diagnostics around `startIdx/maxIdx`, projected transfer bounds,
   and final-dose bbox/per-y support for this case before and after the change.

Risk boundary: widening the transfer launch box can increase runtime, but it
should not invent dose by itself because out-of-field BEV samples remain zero and
the transfer kernels still sample the computed BEV dose. The change must be
validated first with nuclear disabled, then with nuclear enabled, so a primary
support fix is not conflated with halo lattice repair.

### Decisive dose-deficiency dose-diff plan

To attribute the reported deficit:

1. Build with `-DNUCLEAR_CORR=GAUSS_FIT`. Verify `rtdSupportMatrix()` reports halo-capable.
2. Run the same plan twice from Python:
   - Run A: `dose_cal_config["nuclear_correction"] = False` (current default)
   - Run B: `dose_cal_config["nuclear_correction"] = True`
3. Diff B − A. This isolates: (i) the halo contribution itself, (ii) the primary-dose shift from `eRefSq/sigmaDelta/suppressDistalSigmaDip` changing between off and on.
4. Compare Run B to carbonPBS on the same plan. Remaining differences are then narrowed to: shallow off-by-one (`firstMaterial+1`), mass-cutoff entry drops, dose-box z-clip, or H1-H14 numerical hazards. Use the per-layer perf and overflow diagnostics already present to localize.
5. For the shallow-zero specifically, instrument a probe: dump `bevPrimIdd[:, beamFirstInsideRT:beamFirstInsideRT+5]` for one ray to confirm whether the entry slice is being zeroed at the kernel (mass cutoff / firstMaterial+1) or being correctly written but excluded from the dose-box launch.

The end state is that any remaining gap vs carbonPBS is either (a) a documented numerical hazard with a known mitigation, or (b) a contract gap that must be raised on the Python side. Until the rebuild step is done, **no observation about "halo is missing" from a pybind run is informative** — the default config guarantees that outcome.
