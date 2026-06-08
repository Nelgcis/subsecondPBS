## Context

The repository now has two different but related needs that must be handled under one explicit contract:

- `src/tests/wrapper_integration_test.cu` must be able to load `test_data/` directly and call the live RTD wrapper path without going through pybind.
- the compiled Python module under `build/python/cudaCalDose1*.so` must expose not only final-dose entries, but also the additional CarbonPBS-style function family that may later be used by optimization workflows, including CSC and norm related entries.

The user clarified that the direct wrapper test is the primary final-dose path for this change:

- the test should consume the exported `test_data/` variables that correspond to `doseCal.py -> cuCalDose/cuFinalDose` inputs
- the test should feed those values into `raytracedicom_wrapper` or the live wrapper-backed RTD chain directly
- the result of that path is a `dosegrid` output, not a pybind-only compatibility result

At the same time, the shared object must present a broader CarbonPBS-compatible API surface so that later Python-side or integration-side code can import the expected symbols.

Current code facts that create risk:

- `tps_py/dosecal.py` only overrides `cuFinalDose` with the local RTD pybind module, while `cuCalDoseNorm` and `cuCalDose3` still come from the legacy CarbonPBS module.
- the current RTD pybind `cuCalDose3` is not behaviorally equivalent to CarbonPBS; it zeroes CSC outputs and returns a final-dose grid for compatibility only.
- the current RTD wrapper aggregates layer-level spot weights too early for any true spot-wise CSC or row-norm output.
- `wrapper_integration_test` already contains the richest local contract for loading `test_data`, but that contract is still oriented around final-dose verification and does not yet define the full operator-family handoff.

Recent direct-wrapper audit findings also show that final-dose numeric correctness is still a blocking issue:

- the active beam is almost parallel to patient `-x`, so raw dose-array `x` indices cannot be compared naively to physical beam depth.
- the current wrapper audit suggests unresolved risk in the final-dose geometry/sampling chain, especially around:
  - `water_equivalence` depth start and step alignment against RTD fan tracing
  - raw-spot-lattice CPB/ray cropping relative to the projected dose support
  - BEV-to-dose writeback box selection
- the current `[GEOM] Ray grid does not overlap CT volume (XY)` warning is not yet a trustworthy root-cause indicator for this beam geometry and must not be treated as proof of an `x`-flip or raw-bin layout error.

This change therefore needs to do two things without splitting the engine:

- keep one RTD-backed final-dose production path for direct wrapper execution and pybind final-dose calls
- add an explicit operator-family design so the `.so` surface can honestly support, or explicitly gate, the later CSC/norm functions

The same change line now also carries the halo / nuclear-correction recovery work under `NUCLEAR_CORR`.

- halo must remain a second physical branch under the same RTD geometry contract, not a special-case scalar tweak on top of proton dose
- proton-only execution remains the safety baseline; halo may add nuclear dose, but it must not silently perturb the current primary chain
- the current local halo runtime shape is intentionally layer-local to fit the existing wrapper, so the design must define correctness gates before any larger beam-level refactor is considered

## Goals / Non-Goals

**Goals:**

- Make `wrapper_integration_test` the canonical direct-wrapper consumer of `test_data/` for final-dose calculation.
- Ensure the direct wrapper test uses the same exported variables that `doseCal.py -> cuCalDose/cuFinalDose` expects, with no pybind-only hidden preparation.
- Keep one shared CarbonPBS-to-RTD input contract between:
  - `test_data/`
  - `wrapper_integration_test`
  - pybind compatibility entry points
  - `src/core/raytracedicom_wrapper.cu`
- Keep final-dose computation on the existing RTD wrapper-backed path.
- Repair final-dose geometry, sampling, and support correctness before expanding more `.so` operator-family behavior.
- Recover the upstream-faithful halo branch in a way that preserves proton-only semantics when halo is compiled out or runtime-disabled.
- Use one corrected RTD geometry/transport contract for all later modes instead of letting final-dose, CSC/norm, and biological outputs drift into separate geometry paths.
- Expose the broader `.so` API surface needed by downstream code, including:
  - `cuCalDose`
  - `cuFinalDose`
  - `cuCalDoseNorm`
  - `cuCalDose3`
  - `cuFinalDoseAndRBEMap`
  - any explicitly supported fluence-map biological helpers
- Replace fake compatibility behavior with honest semantics:
  - either real implementation
  - or explicit unsupported/runtime-gated behavior
  - but not zero-filled silent substitutes for CSC-style outputs
- Define a phased path for true RTD-native support of CSC and norm outputs.

**Non-Goals:**

- Rewriting `raytracedicom_wrapper` into a different architecture.
- Introducing a second independent dose engine.
- Broad performance tuning unrelated to the direct-wrapper or operator-family contract.
- Speculative smoothing or tuning work before the primary final-dose geometry chain is numerically trustworthy.
- Claiming that all CarbonPBS optimization functions are numerically equivalent before spot-preserving RTD support exists.
- Replacing the current primary path with a halo-first rewrite or a beam-global halo refactor before proton-only and halo-off gates are trustworthy.
- Changing `doseCal.py` production behavior in ways unrelated to contract clarification and backend selection.
- Adding ad hoc coordinate flips or one-off compatibility hacks that are not backed by the RTD/CarbonPBS geometric contract.

## Decisions

### Decision: Treat `wrapper_integration_test` as the direct-wrapper oracle for final dose

The repository already has a direct C++ test harness that can call the live RTD wrapper path. This harness will remain the primary oracle for final-dose contract correctness.

Rationale:

- it removes pybind-only noise when checking whether RTD can consume `test_data`
- it aligns with the user’s clarified intent for the direct-wrapper path
- it anchors future pybind compatibility work on a real non-Python execution path

Alternatives considered:

- drive all validation through pybind first. Rejected because it obscures whether failures come from wrapper physics or from binding-side normalization.

### Decision: Final-dose numeric correctness is the blocking gate for the rest of this change

The broader `.so` operator-family work should not advance on top of an unresolved final-dose geometry chain.

The blocking final-dose areas are:

- beam-aware comparison semantics
- `water_equivalence` depth-start and step alignment
- CPB/ray lateral support and crop bounds
- BEV-to-dose writeback support

Rationale:

- operator-family migration on top of a wrong final-dose geometry baseline would only spread the same error into CSC, norm, and biological outputs
- the current wrapper audit already points to concrete geometry/sampling risks, so fixing those is not speculative cleanup
- this keeps the change focused on necessary correctness work instead of broad uncertain “improvements”

Alternatives considered:

- proceed with operator-family expansion in parallel and reconcile dose correctness later. Rejected because it multiplies the number of moving pieces without establishing a trustworthy baseline.

### Decision: Compare dose in physical and beam-aligned coordinates, not raw array `x` indices

For the current beam, physical propagation is along patient `-x`, while RTD internally uses a fan/gantry coordinate where the relevant depth axis is fan `z`. Final-dose debugging and validation must therefore distinguish:

- patient `x`
- physical `x(mm)`
- beam depth along patient `-x`
- RTD fan `z`
- raw array index order

Rationale:

- this prevents false diagnoses such as “`x`-flip” when the real problem is depth mapping or support truncation
- it makes wrapper-vs-CarbonPBS comparisons stable across different storage layouts and viewers

Alternatives considered:

- continue comparing slice positions by raw array index. Rejected because that is exactly how beam-direction and storage-direction confusion re-enters the debugging loop.

### Decision: Keep one shared CarbonPBS-style input contract for both direct-wrapper and pybind paths

The same exported `test_data` variables should be consumable by:

- the direct wrapper test
- the final-dose pybind entries
- the future operator-family pybind entries

This requires one explicit normalization layer that is shared in meaning even if the actual loader code differs between C++ and pybind.

Rationale:

- avoids one contract for the test harness and another contract for the `.so`
- reduces future drift between `wrapper_integration_test` and Python-facing calls
- makes `test_data` the stable bridge back to `doseCal.py` semantics

Alternatives considered:

- let the test harness keep its own fixture conventions while pybind uses a different interpretation. Rejected because this recreates the current divergence problem.

### Decision: Separate RTD outputs into three modes under one physical chain

The RTD-backed system should be treated as one physical engine with three output modes:

1. final-dose aggregated mode
2. spot-preserving operator mode
3. biological final-map mode

The intended mapping is:

- `cuCalDose` / `cuFinalDose` -> final-dose aggregated mode
- `cuCalDoseNorm` / `cuCalDose3` -> spot-preserving operator mode
- `cuFinalDoseAndRBEMap` -> biological final-map mode

Rationale:

- final dose does not require preserving spot identity
- CSC and row norm fundamentally do require preserving spot identity
- biological final maps should not be forced through CSC unless legacy compatibility truly requires it

Alternatives considered:

- keep pretending `cuCalDose3` is “close enough” if it returns final dose. Rejected because that is semantically false and unsafe for optimization-side code.

### Decision: Do not let the current fake `cuCalDose3` remain as silent compatibility

The current RTD pybind `cuCalDose3` behavior, which zeroes CSC outputs and returns a dose grid, must not remain the long-term contract.

The replacement policy is:

- if the operator mode is implemented, `cuCalDose3` must emit real CSC-compatible outputs
- if the operator mode is not yet implemented, the function must fail explicitly or be clearly capability-gated
- it must not silently return zero CSC buffers as if that were valid

Rationale:

- downstream optimization code cannot safely detect this semantic drift
- zero CSC is worse than an explicit unsupported error because it can produce numerically misleading behavior

Alternatives considered:

- keep the fake behavior for convenience. Rejected because it hides an incompatible operator contract.

### Decision: Direct-wrapper final-dose work comes first, but `.so` operator entries must be planned in the same change line

The first production-grade requirement is:

- direct-wrapper final-dose from `test_data`

But the `.so` surface must be planned and staged in the same design so that later work does not bolt CSC/norm support on as an afterthought.

Rationale:

- this matches the user’s clarified priority
- it keeps the direct-wrapper path grounded while still making room for the broader API surface

Alternatives considered:

- ignore the `.so` function family until final-dose convergence is complete. Rejected because it leaves the required public contract undefined.

### Decision: All RTD modes must reuse one corrected geometry/sampling/crop contract

Once the final-dose geometry chain is repaired, its corrected contract must be reused by:

- direct-wrapper final dose
- pybind final-dose entries
- future spot-preserving operator mode
- future direct biological final-map mode

This means later CSC/norm or biological work must not introduce a second independent mapping for:

- world/patient to gantry/fan coordinates
- `water_equivalence` depth interpretation
- CPB/ray support selection
- BEV-to-dose writeback support

Rationale:

- the repository already risks drift between wrapper and pybind entry paths
- a second geometry path would recreate the same debugging problem under different function names
- this preserves RTD’s core high-efficiency design rather than forking the physics chain

Alternatives considered:

- let operator mode and biological mode build their own geometry handling for convenience. Rejected because it would duplicate the most error-prone part of the system.

### Decision: Add a dedicated spot-preserving RTD operator mode rather than reusing the already-aggregated final-dose intermediate

The current wrapper path rasterizes and aggregates spot weights per layer too early for CSC-style outputs. The fix is not to infer CSC from final dose after the fact. The fix is to introduce a dedicated operator mode that preserves spot identity through the relevant transport stage.

Rationale:

- `cuCalDoseNorm` and `cuCalDose3` are spot-operator outputs, not final-dose aliases
- the current aggregated layer flow cannot recover per-spot coefficients once spot identity has been merged

Alternatives considered:

- derive CSC from final-dose grids or from post-aggregation layer intermediates. Rejected because the needed spot dimension has already been destroyed.

### Decision: Biological map functions should be split into direct-final-map and CSC-derived families

Not all biological helpers should be treated the same:

- `cuFinalDoseAndRBEMap` should be designed as a direct final-map RTD extension
- CSC-dependent helpers such as fluence-map biological postprocessing may remain legacy-style only until real operator mode exists

Rationale:

- direct final biological maps are closer to final-dose accumulation than to CSC export
- forcing everything through CSC would make the RTD design less natural and more memory-heavy

Alternatives considered:

- require every biological map function to wait for CSC first. Rejected because it would delay direct final-map support unnecessarily.

### Decision: Do not use CarbonPBS-only profile or subspot corrections to mask primary geometry defects

Profile/subspot terms may later help RTD match CarbonPBS lateral smoothness more closely, but they are explicitly second-order work. They must not be used as a substitute for fixing:

- depth-start mismatch
- wrong step/count interpretation
- incorrect lateral support crop
- incorrect BEV-to-dose writeback bounds

Rationale:

- these corrections can change smoothness, but they do not honestly explain hard support truncation or gross proximal/distal misalignment
- adding them early would risk hiding the real defect and make the baseline harder to trust

Alternatives considered:

- enable more CarbonPBS-specific smoothing first and see whether the dose “looks better”. Rejected because appearance improvement without geometry correctness would be misleading.

### Decision: Keep halo as a gated sibling branch on top of the stable proton path

Halo recovery must preserve the existing primary chain as the reference behavior. The local wrapper may stay layer-local for now, but halo must be treated as a second branch with separate nuclear buffers and explicit accumulation into the shared dose volume.

The practical stage split is:

- shared upstream-of-IDD work: density tracing, WEPL, entry/cutoff determination, primary ray setup, and the current primary superposition/transfer path
- halo-only additions: nuclear PB mapping, dual-output nuclear IDD/sigma, nuclear superposition, and `nucTransfDiv`

Rationale:

- this matches upstream RTD's mental model more closely than a scalar correction approach
- it preserves a clean proton-only safety baseline while allowing layer-local integration in the local wrapper
- it keeps correctness questions local to the halo branch instead of reopening the already fragile primary chain

Alternatives considered:

- rewrite the wrapper around a beam-global halo-first structure immediately. Rejected because the current local beam orchestration is already complex and primary correctness is still the hard gate.

### Decision: Halo enablement must follow an explicit A/B/C gate matrix

Halo work must not be evaluated as one undifferentiated "on/off" feature. The required gate order is:

- gate A: `NUCLEAR_CORR=OFF` build baseline
- gate B: `NUCLEAR_CORR` enabled build with `nuclear_correction=false`
- gate C: `NUCLEAR_CORR` enabled build with `nuclear_correction=true`

The acceptance rule is:

- A and B must match before C is used to judge halo physics
- C is only meaningful after A/B prove that the compiled halo path leaves proton-only behavior unchanged

Rationale:

- this cleanly separates "compiled branch perturbs primary" from "halo physics mismatches reference"
- it gives the existing `wrapper_integration_test` halo audit a precise role instead of a generic diagnostic role
- it prevents halo tuning from masking regressions in the primary path

Alternatives considered:

- compare halo-enabled output directly against a reference and debug from there. Rejected because a mismatch could come from either proton drift or halo physics, which is too ambiguous.

### Decision: Investigate halo mismatches in geometry-first order, not superposition-first order

The most likely remaining local halo differences are not in the generic superposition helper itself, but in how the halo lattice and nuclear transfer geometry are mapped onto the current wrapper coordinates.

The required triage order is:

- `HaloLatticePlan` / `devNucSpotIdx` equivalence against upstream spot-grid semantics
- `nucRayIdxToDoseIdx` / `nucTransfDiv` support-box equivalence against upstream transfer geometry
- only after those pass, consider whether the local nuclear superposition launch wrapper must be replaced with a more literal upstream launch site

Rationale:

- the local `performCompleteTileBasedSuperposition(...)` path still wraps `tileRadCalc` plus batched `kernelSuperposition<r>` launches, so it is closer to upstream behavior than its helper name suggests
- halo support shifts and centering errors are more plausibly introduced by spot-grid mapping or transfer geometry than by the already-restored superposition kernel family
- this ordering minimizes unnecessary runtime churn

Alternatives considered:

- replace the local superposition helper first for maximum literal upstream parity. Rejected because it is higher-cost and not yet the highest-probability defect source.

## Architecture Sketch

```text
test_data/
  -> shared CarbonPBS-style variable contract
  -> two consumers

  A. direct wrapper test
     wrapper_integration_test
     -> RTDBeamSettings / RTDEnergyStruct
     -> subsecondWrapper(...)
     -> dosegrid

  B. pybind compatibility layer
     cudaCalDose1.so
     -> shared normalization rules
     -> function family

        final-dose family
        -> cuCalDose / cuFinalDose
        -> subsecondWrapper-backed final dose

        operator family
        -> cuCalDoseNorm / cuCalDose3
        -> spot-preserving RTD operator mode

        biological family
        -> cuFinalDoseAndRBEMap
        -> direct RTD biological accumulation

        optional legacy-derived family
        -> CSC-based biological helpers
        -> only after real operator mode exists
```

## Implementation Shape

### Shared contract layer

The design assumes one explicit contract table for the following exported variables:

- `water_equivalence`
- `roiIdx` or canonical ROI normalization equivalent
- `all_energies`
- `source_pos`
- `beam_dir`
- `beam_xdir`
- `beam_ydir`
- `doseGrid_corner`
- `doseGrid_resolution`
- `doseGrid_dims`
- `longitudal_cutoff`
- `energy_list`
- `idd_data`
- `idd_setting`
- `profile_data`
- `profile_setting`
- `beam_para_data`
- `subspot_data`
- `layer_info`
- `layer_energy`
- `idbeamxy`
- `number_particle` or any final-dose-only weight field

This contract must clearly distinguish:

- what is required for direct final dose
- what is required for spot operators
- what is required for biological maps

### Final-dose path

The final-dose path stays on the existing RTD wrapper-backed chain.

Required property:

- `wrapper_integration_test` and pybind `cuFinalDose/cuCalDose` must be two boundary adapters onto the same RTD-backed final-dose semantics

Required gate before broader migration:

- final-dose output must be correct in support extent, proximal tail, distal tail, and XY continuity relative to the accepted CarbonPBS/reference comparison
- the corrected baseline must be defined in physical and beam-aligned coordinates, not only by raw index-space image inspection

Current focus areas within that gate:

- depth-axis alignment between RTD fan tracing and exported `water_equivalence`
- preservation of spot-lattice phase and CPB/ray support
- correctness of BEV-to-dose writeback coverage

### Operator path

A new operator path is required for:

- `cuCalDoseNorm`
- `cuCalDose3`

Its internal requirements are:

- preserve spot identity through batch processing
- support ROI-only accumulation
- support unweighted per-spot coefficients
- support count-first then write-second CSC generation
- support row-norm accumulation without materializing full CSC when possible
- reuse the corrected final-dose geometry/sampling/crop contract instead of introducing a second mapping path

### Biological path

A direct RTD biological path is required for:

- `cuFinalDoseAndRBEMap`

This path should share the same geometric and transport normalization as final dose, while extending accumulation outputs for biological maps.

CSC-based biological helper functions should not be declared RTD-native until the operator path exists.

### Halo path

The current local halo path should be understood as a layer-local sibling branch whose runtime stages are:

```text
shared tracing / entry state
  -> proton fillIddAndSigma outputs
  -> primary superposition
  -> primTransfDiv

  plus, when halo is compiled and enabled:
  -> HaloLatticePlan / devNucSpotIdx
  -> nuclear fillIddAndSigma outputs
  -> nuclear superposition
  -> nucTransfDiv
  -> additive write into the same dose volume
```

Current accepted local divergence from upstream:

- local orchestration is layer-local instead of beam-global
- the nuclear superposition launch site is wrapped by `performCompleteTileBasedSuperposition(...)` instead of being inlined in the wrapper, but it still follows the `tileRadCalc + kernelSuperposition<r>` structure
- nuclear LUTs may be resampled onto the active primary depth axis before texture binding when the imported CarbonPBS-backed axis differs from the reference RTD axis

Required halo gates before TPS enablement:

- prove gate A/B proton invariance first
- prove halo-on support extent and lateral tail behavior on at least one controlled case
- keep halo disabled for TPS if Section 0 final-dose geometry remains numerically unresolved, even if the branch compiles

## Risks / Trade-offs

- [Final-dose direct-wrapper work and operator-family work can become entangled] -> keep the direct-wrapper final-dose loader and the operator-mode design distinct, even though they share the same input contract.
- [Work on CSC/norm may restart before final-dose geometry is actually trustworthy] -> make final-dose correctness an explicit blocking gate and require later work to consume the repaired baseline.
- [The `.so` surface may expose functions before the RTD operator mode is ready] -> use explicit unsupported behavior instead of fake zero-output compatibility.
- [Adding spot-preserving mode can increase memory pressure] -> design operator mode around spot batches and ROI-restricted output rather than full dense spot-by-voxel tensors.
- [Biological helpers can drift into a separate implicit engine] -> require them to share the same normalization and physical chain assumptions as final dose.
- [`test_data` naming or export conventions may keep changing] -> keep loader rules metadata-driven and explicit, not filename-guess-driven only.
- [Direct wrapper and pybind loaders can still diverge in corner cases] -> define contract checks and invariants in both paths and keep them auditable.
- [Speculative profile/subspot tuning could hide the real defect] -> defer second-order smoothing until depth/support/writeback correctness is closed.
- [A second geometry path could emerge inside operator or biological migration] -> explicitly ban duplicated geometry/sampling implementations and require reuse of the repaired final-dose contract.
- [Halo-on mismatches could be blamed on the wrong stage] -> require a geometry-first triage order: halo lattice mapping, then nuclear transfer geometry, then only later superposition launch structure.
- [Compiled halo support could silently perturb proton-only behavior] -> require the A/B/C gate matrix and treat A/B invariance as a hard prerequisite for any halo reference comparison.
- [Halo recovery could consume performance effort before correctness is known] -> defer Section 8 reprioritization until halo-extra workspace and transfer-box deltas are measured on the current runtime path.

## Migration Plan

1. Lock one beam-aware comparison vocabulary so patient `x`, physical `x(mm)`, beam depth along patient `-x`, RTD fan `z`, and raw array index order are not mixed during validation.
2. Make `wrapper_integration_test` consume `test_data` as the canonical direct final-dose path.
3. Repair final-dose geometry/sampling correctness on the existing RTD wrapper-backed chain, concentrating on:
   - `water_equivalence` depth-start and step alignment
   - raw-spot-lattice phase preservation and CPB/ray support
   - BEV-to-dose writeback coverage
4. Freeze one corrected final-dose regression baseline for direct-wrapper execution.
5. Align pybind final-dose entries to that same corrected normalized contract and remove any hidden drift.
6. Replace fake `cuCalDose3` semantics with honest capability behavior.
7. Add spot-preserving operator infrastructure on top of the corrected RTD geometry/transport chain.
8. Implement `cuCalDoseNorm`.
9. Implement `cuCalDose3`.
10. Add direct biological final-map support.
11. Revisit CSC-dependent biological helpers only after the operator path is real.
12. Freeze the halo A/B/C gate matrix and connect all halo comparisons to one auditable reporting surface.
13. Prove that halo does not alter upstream-of-IDD tracing, entry/cutoff state, or proton-only transfer geometry when runtime-disabled.
14. Audit `HaloLatticePlan` and `devNucSpotIdx` equivalence before changing the nuclear superposition launch shape.
15. Audit `nucRayIdxToDoseIdx` and `nucTransfDiv` support geometry before changing the nuclear superposition launch shape.
16. Only if halo reference mismatches remain after steps 14 and 15, revisit whether the local nuclear superposition wrapper must be replaced with a more literal upstream launch site.
17. Feed measured halo workspace and support-box deltas back into the Section 8 performance plan.

## Open Questions

- Is `weqHeader[0]` in the current fixture exactly the projected beam-depth start expected by CarbonPBS for this exported `water_equivalence`, or is there still an additional convention offset that must be normalized explicitly?
- Should the raw-spot-lattice CPB/ray support always include the full projected dose support, or should there be a formally defined beam-support crop rule that is wider than spot-bounds-plus-sigma but still cheaper than full-volume support?
- Should the first operator-family milestone export explicit unsupported errors for CSC-related functions, or should those functions remain unbound until real support lands?
- Do any downstream callers currently rely on the fake RTD `cuCalDose3` returning `dose_grid`, or can that behavior be removed cleanly?
- Is `source_pos` required to remain beam-level only for the direct final-dose path, while future CSC-derived biological helpers need a true spot-wise source contract?
- Which biological helper functions are truly required in the first `.so` milestone, and which can remain explicitly deferred after `cuCalDoseNorm` and `cuCalDose3`?
- Is the current `HaloLatticePlan` rounding and `approxEq(...)` tolerance truly equivalent to upstream spot-grid placement on all production beams, or does it still hide a half-cell centering mismatch?
- Should the existing halo gate thresholds remain purely numeric (`MAE` / `RMSE`), or should support-extent deltas become first-class pass/fail criteria once a controlled halo reference is fixed?
