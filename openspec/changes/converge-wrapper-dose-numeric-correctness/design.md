## Context

The repository now has a connected wrapper-backed main path:

- `tests/wrapper_integration_test` can build realistic fixture inputs and execute `src/core/raytracedicom_wrapper.cu` directly.
- the public pybind entries can reach the same wrapper-backed chain.
- recent work fixed pathing, output-contract, and callability issues so the remaining problem is no longer “can the code run?” but “does the code produce the right dose?”

The observed failure modes are numerical and stage-coupled:

- Bragg peak appears at the wrong depth.
- dose distribution shape is abnormal or collapses in depth.
- energy layers may be mapped to the wrong LUT or physical depth behavior.
- final dose writeback may still be transposed, reversed, or otherwise inconsistent with the intended depth axis.

The user constraint remains strict: keep `raytracedicom_wrapper` in its current basic form, do not add a parallel implementation, do not turn this into a broad refactor, and use `tests/wrapper_integration_test` as the main debugging baseline.

## Goals / Non-Goals

**Goals:**

- Converge the existing wrapper-backed main path toward numerically correct final dose output using `tests/wrapper_integration_test` as the primary reference harness.
- Debug and correct the chain stage by stage: CPB geometry, reference-plane geometry, WEPL generation, LUT lookup, sigma transport, IDD/CIDD handling, superposition, and final dose-grid writeback.
- Require each stage to produce a concrete error-source summary and a correction summary rather than mixing multiple unverified adjustments together.
- Keep the public pybind final-dose path numerically aligned with the same corrected wrapper-backed path.

**Non-Goals:**

- Rewriting `raytracedicom_wrapper` into a different architecture.
- Replacing the current RTD tracing or dose pipeline with a second implementation.
- General performance optimization, kernel tuning, or unrelated cleanup work.
- Declaring numerical success through heuristic smoothing or post-processing that hides stage-level errors.
- Returning a finite-but-degraded dose through silent fallback behavior that no longer represents the RayTraceDicom prototype or the CarbonPBS exported inputs.

## Decisions

### Decision: Use `tests/wrapper_integration_test` as the stage-by-stage oracle

All numerical debugging will anchor on the existing wrapper fixture path first, because it exposes the most direct view of the wrapper contract and already contains auditing, summaries, and reference-dose comparison hooks.

Rationale:

- The wrapper test isolates numerical problems from pybind-only callability concerns.
- It already writes output artifacts and prints comparison summaries against repository reference data.

Alternatives considered:

- Start from Python only. Rejected because it adds boundary noise before the internal numerical chain is stable.

### Decision: Split convergence into explicit pipeline stages

The change will treat the numerical chain as a sequence of independently inspectable stages:

1. CPB and reference-plane geometry
2. WEPL and entry-plane transport
3. LUT and energy-layer mapping
4. sigma transport and IDD/CIDD sampling
5. superposition and accumulation
6. final dose-grid writeback and depth orientation

Each stage must identify:

- expected invariant
- observed failure mode
- likely error source
- confirmed correction
- downstream effect after correction

Rationale:

- Bragg peak shifts and abnormal dose shapes often come from compounded upstream and downstream errors.
- Sequential isolation is lower risk than changing several coupled steps simultaneously.

Alternatives considered:

- Apply broad empirical fixes across multiple stages at once. Rejected because it makes the final cause of improvement untraceable.

### Decision: Keep corrections local and in-place

Numerical corrections should modify the existing wrapper-backed path in place and keep the current control flow recognizable. Instrumentation may be added or refined, but the design does not authorize a major restructuring of the wrapper or a replacement of its stage boundaries.

Rationale:

- The user explicitly wants the wrapper basic form preserved.
- A local correction strategy reduces contract drift versus the newly completed pybind path.

Alternatives considered:

- Extract large new helpers or alternate branches for “corrected” behavior. Rejected because it risks splitting the production path again.

### Decision: Prototype fidelity beats degraded fallback

For the CarbonPBS compatibility entry, the wrapper is only allowed to do one of two things:

1. preserve the physical meaning of the exported plan inputs and the intended RayTraceDicom prototype chain, or
2. stop with an explicit diagnostic that identifies which contract could not be honored.

It must not silently switch to a numerically convenient but semantically different path merely to produce a finite dose array.

This specifically forbids the following classes of behavior unless they are first proven equivalent to the prototype path:

- replacing real spot maps with synthesized or layer-only surrogate weights
- dropping real spot positions or particle counts during fallback
- silently skipping malformed beams and continuing dose accumulation
- silently stretching or reinterpreting units when exported metadata is ambiguous
- keeping CarbonPBS-specific inputs only “for audit” while running a different physical chain without an explicit unsupported-input decision

Rationale:

- The user requirement is strict: no “为了出结果就写死或者掩盖问题”.
- A tiny-but-finite dose is more dangerous than a hard failure because it can look numerically plausible while violating plan physics.

Alternatives considered:

- Keep degraded fallback paths for convenience while logging warnings. Rejected because warnings do not prevent silent numerical misuse.

### Decision: Zero-weight spots are valid no-dose inputs, not malformed geometry

Zero-particle or zero-weight spots exported by CarbonPBS remain part of the plan lattice contract. They should not invalidate lattice reconstruction by themselves. The wrapper may ignore them as no-dose contributors, but it must still preserve the positive-weight spot map exactly.

Rationale:

- Current fixtures already contain many zero-particle spots in otherwise valid plans.
- Treating a zero-weight spot as a beam-level fatal geometry error forces the wrapper into degraded fallback behavior that discards the real spot map.

Alternatives considered:

- Continue treating any non-rasterized spot as a lattice-construction failure. Rejected because it confuses valid no-dose spots with malformed positive-dose inputs.

### Decision: Ambiguous geometry or unit inference must fail fast

Heuristic recovery remains acceptable only as an audit aid during exploration. Production convergence for this change requires that ambiguous geometry or unit contracts become explicit: either the exported data proves the intended units and geometry, or the wrapper must require an override / reject execution.

Examples of invariants that must hold before the relevant kernels execute:

- finite positive `sad` / `sourceDist`
- finite `startZ` and `entryZ`
- strictly positive `pxSpMult` / effective `pixelSp`
- physically valid `energyDepthToMm` and cutoff normalization
- non-negative `sigmaSq`, finite `rSigmaEff`, and positive `stepVol` in the RTD physical chain

Rationale:

- The current failure class “NaN/Inf and mid-pipeline truncation” is most consistent with invalid geometry or unit propagation rather than with a merely empty dose.
- Catching non-physical states before superposition is safer and more explainable than detecting NaN only in the final dose grid.

Alternatives considered:

- Leave heuristics in place and rely on final host-side NaN checks. Rejected because it obscures the first failing stage and encourages silent drift.

### Decision: Sliced LUT exports require explicit semantics, not reconstructed full axes

The current pybind compatibility layer still reconstructs sliced CarbonPBS LUT exports heuristically:

- `select_slice_energies(...)` picks retained energy rows by assuming a simple head/tail slice
- `profileSetting[1] *= logical_depth_n/profile_depth_n` stretches the exported profile depth step to span the original logical depth count

Those moves are only valid if the exporter proves a uniform-decimation slice contract with matching energy-row selection. If the export is instead a tail crop, head crop, or any other compact representation, then the wrapper is feeding the RTD chain the wrong physical depth coordinates and wrong energy-row alignment.

Production convergence for this change therefore requires one of:

1. explicit metadata that proves the sliced export can be reconstructed exactly, or
2. full-depth exports, or
3. an explicit fail-fast rejection for unsupported sliced exports.

Rationale:

- The current `bg838`-style peak-shape distortion is not explained by magnitude loss alone; depth-axis reconstruction drift can also sharpen or mis-shape the profile.
- The same plan data can appear “sometimes okay” or “sometimes broken” depending on how the sliced export happens to line up with the current heuristics.

Alternatives considered:

- Keep the current slice reconstruction and only log warnings. Rejected because it leaves the physical depth contract ambiguous.

### Decision: Virtual source distance may be inferred only when the plan geometry proves it

For the CarbonPBS `cuFinalDose` entry, pybind currently initializes `sourceDist=(0,0)` and relies on wrapper-side recovery from spot directions when possible. That can be acceptable only if the exported spot directions and departure-plane positions jointly determine a consistent `sadx/sady` estimate.

If that estimate is missing, unstable, or inconsistent across spots, the wrapper must reject the beam rather than silently falling back to scalar `sad` for divergent fan geometry.

Rationale:

- `bg838` itself showed that inferred source distance can be correct, so this path is not inherently invalid.
- The `NaN/Inf` class is still consistent with plans where the inferred or fallback source distance makes `pxSpMult`, `pixelSp`, or later sigma geometry non-physical.

Alternatives considered:

- Always fall back to scalar `sad` when `sourceDist` is absent. Rejected because scalar SAD is not a prototype-equivalent replacement for divergent virtual source distances.

### Decision: Treat pybind as a consumer of the corrected chain, not a separate tuning target

Numerical convergence will be established on the direct wrapper fixture path first, then confirmed through pybind on the same corrected path. Pybind may still need shape or writeback fixes where they directly affect numerical interpretation, but it is not the primary location for transport or depth-dose tuning.

Rationale:

- The numerical chain lives in the wrapper path.
- Python-facing verification is still required, but it should validate the same corrected behavior rather than drive a separate tuning process.

Alternatives considered:

- Tune pybind-visible output independently of the wrapper fixture. Rejected because it hides the real source of error.

### Decision: Make stage-wise error-source reporting a first-class output

The implementation and validation plan will explicitly produce a correction summary per stage, including what was wrong, how it was detected, what was changed, and how the dose error moved afterward.

Rationale:

- The user explicitly requested “每阶段的误差来源与修正摘要”.
- This creates a defensible debugging record instead of undocumented trial-and-error.

Alternatives considered:

- Only report the final corrected result. Rejected because it would obscure the numerical reasoning and make regression harder to diagnose later.

## Risks / Trade-offs

- [Multiple coupled errors can mask one another] → Fix and validate one stage at a time, preserving per-stage summaries and re-running fixture comparisons after each correction.
- [CUDA environment instability can block runtime confirmation] → Separate environment-only failures from numerical failures and avoid claiming convergence without actual stage output when the runtime cannot execute.
- [Instrumentation noise can become large and intrusive] → Prefer focused summaries and stage checkpoints over uncontrolled logging.
- [Writeback-order fixes may appear numerically beneficial while hiding upstream geometry errors] → Do not accept output-only corrections without checking upstream CPB, WEPL, LUT, and IDD invariants first.

## Migration Plan

1. Use `tests/wrapper_integration_test` and current reference-dose artifacts to establish the present failure signature.
2. Eliminate silent degraded behavior first: classify fallbacks and skips, preserve prototype-fidelity inputs where possible, and convert non-equivalent fallback branches into explicit runtime failures.
3. Inspect and correct the geometry and reference-plane stages, because depth-direction and Bragg peak placement are most sensitive to upstream coordinate errors.
4. Move to WEPL, LUT, and IDD/CIDD alignment, validating depth-dose behavior after each correction.
5. Correct sigma transport and superposition only after the physical depth axis and energy mapping are stable.
6. Finish by validating dose writeback orientation and pybind-facing dose-grid equivalence.
7. Record per-stage error-source and correction summaries, then confirm the same corrected behavior from Python.

## Confirmed Failure Classes

The current investigation has already separated the observed production regressions into two distinct classes. The implementation plan must treat them differently.

### Failure Class A: tiny but finite dose caused by degraded fallback

Representative case: `bg838_beam964`.

Confirmed characteristics:

- the plan contains many valid zero-particle spots
- `buildRawSpotLattice` currently fails on zero-weight spot rasterization instead of treating those spots as ignorable no-dose inputs
- the beam then falls into the current `legacy CPB fallback path`
- that fallback reconstructs CPB weights from `subspotData` only and drops the real per-spot particle counts / positions

Resulting symptom:

- output remains finite but is lower than the intended beam magnitude by many orders of magnitude and the lateral field shape collapses toward a much smaller surrogate support

Design implication:

- this is not an acceptable “best effort” execution mode
- the fallback path must be removed, proven equivalent, or replaced by fail-fast behavior

### Failure Class B: NaN/Inf caused by invalid physical state propagation

Representative cases: plans that previously terminated with `cuFinalDose/calcDose/cuCalDose produced NaN/Inf dose output`.

Most likely contributing chain:

- geometry and unit inference (`startZ`, `entryZ`, `sourceDist`, `energyDepthToMm`)
- raw-lattice convolution scaling (`pxSpMult`, `pixelSp`)
- RTD sigma / IDD transport (`sigmaSq`, `rSigmaEff`, `voxelWidth`, `stepVol`)
- downstream superposition consuming already-invalid per-ray buffers

Resulting symptom:

- execution progresses part-way through the RTD chain and then contaminates later buffers with non-finite values, after which the pybind host-side dose summary aborts on NaN/Inf

Design implication:

- these plans need stage-local invariants and sentries, not a post hoc final-dose-only error
- each stage must fail at the first non-physical state rather than letting NaN propagate

## Current Branch Classification

This matrix captures the currently relevant host-side branches that can change physical meaning or hide input-contract failures. It is intentionally stricter than generic “warning only” auditing. Local in-kernel branches that merely skip zero-dose work while preserving the same physical contract are not the target here.

| Branch / adaptation | Current location | Current behavior | Classification | Required action |
| --- | --- | --- | --- | --- |
| Raw spot lattice built from `layerSpotCounts + spotPositions + spotWeights + WEQ header` | `src/core/raytracedicom_wrapper.cu` | Reconstructs a dense departure plane from exported CarbonPBS spots | Prototype-equivalent candidate | Keep only after acceptance audit proves positive-weight spot counts and sums are preserved |
| Zero-weight spot causes lattice build failure | `buildRawSpotLattice` / `rasterizeContinuousSpotToDensePlane` | Treats zero-written rasterization as beam-level failure | Not allowed | Ignore zero-weight spots as valid no-dose inputs; still fail on malformed positive-weight spots |
| Legacy CPB fallback when `hasRawSpotLattice == false` | `src/core/raytracedicom_wrapper.cu` | Replaces real spot map with layer-level subspot surrogate and returns finite dose | Must become hard runtime error | Remove, hard-disable, or prove prototype equivalence before reuse |
| Missing or malformed beam-critical inputs handled via beam-loop `continue` | `src/core/raytracedicom_wrapper.cu` | Logs an error and silently skips the beam | Must become hard runtime error | Surface beam index and violated contract to caller immediately |
| `sourceDist=(0,0)` with wrapper-side inference from spot directions | `src/bindings/raytracedicom_pybind.cpp` + `src/core/raytracedicom_wrapper.cu` | Accepts missing explicit virtual source distances and tries to recover them from geometry | Requires explicit exported metadata or a consistency proof | Add inference audit; reject beam if recovered `sadx/sady` is not stable and physical |
| `lenToMm` and `energyDepthToMm` heuristics | `src/core/raytracedicom_wrapper.cu` | Infers mm/cm from CT spacing and peak-depth magnitude | Requires explicit metadata or fail-fast | Keep only as debug audit; production path must reject ambiguous units |
| Sliced LUT reconstruction via `select_slice_energies(...)` and stretched `profileSetting` step | `src/bindings/raytracedicom_pybind.cpp` | Reconstructs missing profile/energy axes heuristically | Requires explicit exported metadata | Require slice semantics from exporter or full-depth tables; otherwise reject |
| Per-spot cutoff collapsed to one value per layer | `src/bindings/raytracedicom_pybind.cpp` | Uses first spot in layer and only warns on variation | Allowed only under layer-constant contract | Reject within-layer cutoff variation until a prototype-equivalent mapping is defined |
| Carbon profile sigma / overallWeight / subspot behavior disabled while cutoff is replaced by `BP_DEPTH_CUTOFF * peakDepth` | `src/core/raytracedicom_wrapper.cu` | Runs RTD chain with CarbonPBS-specific modifiers ignored or substituted | Must become explicit unsupported-input decision | Either map these inputs through a proven equivalent transformation or fail-fast |

## Initial Wrapper-Only Audit

This section records the first wrapper-only audit pass for `bg800_beam925` using
`build/bin/wrapper_integration_test`, the fixture CSV inputs under `test_data`,
the current wrapper implementation, `carbonPBS/cudaCalDose.cpp`,
`carbonPBS/deviceCalDose.cu`, and `RayTraceDicom-main`.

### Current Executable Failure Signature

- The current local runtime is blocked by environment, not by a numerically proven wrapper result:
  `cudaMalloc3DArray failed: CUDA driver version is insufficient for CUDA runtime version`.
- Under this environment-only failure, the produced wrapper dose is all-zero and the reference comparison reports:
  `sum(test/ref)=0 / 74681.9`, `max(test/ref)=0 / 0.845188`, `peakSlice test/ref=-1 / 100`.
- Because the run aborts before valid GPU output is produced, downstream stage correctness cannot be confirmed from this host alone. Static contract and code-path auditing therefore takes precedence until a compatible CUDA runtime is available.

### Stage-Wise Debugging Worksheet

| Stage | Current observation | Strictly supported conclusion | Priority |
| --- | --- | --- | --- |
| Wrapper entry / beam contract | `numLayers=29`, `totalSpots=6967`, `layerCutoffs=29`, `spotPositionsAreIndices=1`, WEQ header present | Basic beam/spot array cardinalities are internally consistent | High |
| WEQ / spot coordinate semantics | `dosecal.py` exports `idbeamxy = physical_index + offset + 0.5`; wrapper decodes with `(raw - 0.5) * step` | Wrapper's centered-texel interpretation matches CarbonPBS texture sampling semantics | High |
| Layer cutoff contract | Fixture `longitudal_cutoff` is constant within each layer | Current layer-level cutoff collapse is not losing information for this fixture | High |
| IDD depth axis | Export metadata says logical shape `(123, 4000)` but current fixture provides `slice` export with max observed column `79` | The current fixture does not contain the full IDD depth axis; any reconstruction beyond the exported slice is approximate unless exporter semantics are supplied | Critical |
| Profile depth axis | Export metadata says logical shape `(123, 800, 11)` but current fixture provides `slice` export with max observed indices `[79,39,10]` | The current fixture does not contain the full profile table; current wrapper/test logic can only approximate the original CarbonPBS table semantics | Critical |
| Energy / peak-depth mapping | Input audit shows high-energy rows clamp to `peakDepth=319.213` while energies extend to `399.92` | The current wrapper/test path is synthesizing peak depths from reference tables instead of receiving original CarbonPBS peak-depth data | Critical |
| Cutoff -> RTD range mapping | Wrapper maps `layerCutoffMm` to `peakDepth = layerCutoff / BP_DEPTH_CUTOFF` and then uses RTD `afterLastStep` logic | This is a deliberate semantic adaptation, not a proven equivalence to CarbonPBS's direct `weqDepth < longitudalCutoff` rule | Critical |
| SAD / source-distance inputs | `dosecal.py` passes scalar `sad`, common `sourcePos`, per-spot `beamDir`; wrapper infers `sourceDist` from spot directions when needed | No additional `dosecal.py` field is strictly required here for this fixture; the missing semantics are in table depth axes, not source geometry | Medium |

### Confirmed Contract Facts From CarbonPBS And `dosecal.py`

- `dosecal.py` exports `idbeamxy[:,0] = x_all + xlim + 0.5` and `idbeamxy[:,1] = z_all + ylim + 0.5`, while `rayweq[3:9] = [-ylim, 1, 2*ylim+1, -xlim, 1, 2*xlim+1]`.
- `carbonPBS/deviceCalDose.cu` samples WEQ and profile directly in texture coordinates using that centered `idbeamxy` convention.
- `carbonPBS/cudaCalDose.cpp` resolves `energyIdx` per layer via `binarySearchEneIdx(layerEnergy, enelist, nEne)`.
- `carbonPBS/cudaCalDose.cpp` also takes `longitudalCutoff` from the first spot in the layer. The current fixture is layer-constant, so that legacy behavior is not losing information here.
- `dosecal.py` does not provide explicit per-layer peak depths; the CarbonPBS runtime does not require them. The wrapper/test path currently invents them from external reference tables.

### Data That Is Actually Missing Versus Data That Can Be Inferred

Missing and needed if strict CarbonPBS-to-wrapper equivalence is required:

- The full unsliced `idd_data` depth axis, or an explicit exporter reconstruction rule for `export_mode=slice`.
- The full unsliced `profile_data` depth axis, or an explicit exporter reconstruction rule for `export_mode=slice`.
- Original CarbonPBS peak-depth semantics if the wrapper is expected to behave like RTD main without using the current cutoff-derived surrogate.

Not currently missing for this fixture and should be inferred from existing contracts:

- Per-spot cutoff behavior: the fixture is layer-constant.
- Separate `sadx` / `sady`: the legacy CarbonPBS final-dose runtime itself only consumes scalar `sad` plus spot directions.
- Per-spot source positions: the legacy final-dose runtime consumes a common source position plus per-spot beam directions.

### Immediate Implication For Subsequent Tasks

- Geometry and WEQ auditing can proceed using existing fixture data and CarbonPBS semantics.
- LUT / IDD / profile convergence is blocked from being strictly proven until the sliced table semantics are either reconstructed from the exporter or replaced with full-depth exports.
- Any numeric correction to `peakDepth`, `afterLastStep`, or sigma that ignores the sliced-table limitation should be treated as provisional.

## Open Questions

- What quantitative acceptance thresholds should be treated as “converged enough” for Bragg peak position, peak amplitude, and dose-shape comparison against the current fixture reference?
- Which current fixture discrepancies are known physics-model differences versus actual integration mistakes?
- Does any remaining writeback mismatch reflect storage-order error only, or also an upstream depth-axis interpretation problem?
- Can the exporter provide the full unsliced `idd_data` and `profile_data`, or document the exact reconstruction semantics intended for `export_mode=slice`?
