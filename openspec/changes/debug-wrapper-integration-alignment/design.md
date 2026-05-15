## Context

The repository now has two contract-oriented OpenSpec changes, but neither one proves where the live wrapper path diverges from the test harness that is supposed to feed it. The current debugging target is narrower and more execution-oriented:

- `src/tests/wrapper_integration_test.cu` already loads beam settings, energy data, WEQ-related metadata, LUT-backed tensors, and authoritative geometry overrides from `calc_required_meta.csv`
- that test then calls the same `subsecondWrapper(...)` entry used by the current integration path
- `src/core/raytracedicom_wrapper.cu` contains several normalization and inference stages that can shift semantics before dose is accumulated
- current hot spots already visible in source include beam-basis sign correction, `sourceDist` fallback, multiple `idbeamxy` decode formulas, WEQ-to-BEV transfer, `energyDepthToMm` scaling, layer-cutoff conversion, `profileRowIdx` selection, and `subspotData` row remap or flattening

The user constraint is explicit: keep the wrapper basic form, do not add a parallel implementation, and do not refactor the main flow. The change therefore needs to add observability and comparison inside the current path rather than replace the path.

## Goals / Non-Goals

**Goals:**

- Use `src/tests/wrapper_integration_test.cu` as the baseline input oracle for wrapper debugging.
- Add stage-by-stage trace points inside the existing wrapper path so the repository can compare external inputs with the wrapper's normalized internal state.
- Localize concrete mismatch points across geometry, coordinate frames, grids, ROI, energy layers, spot or subspot mapping, LUT row selection, and convolution setup.
- Produce a ranked diagnosis of the mismatch stages most likely responsible for abnormal dose and Bragg peak displacement.
- Preserve the wrapper's single execution path and current numerical behavior while diagnostics are added.

**Non-Goals:**

- Building a second wrapper implementation for comparison.
- Refactoring the wrapper into a different control-flow shape before the mismatch is proven.
- Changing physics kernels, convolution math, or beam-model behavior as part of this change.
- Declaring unproven equivalence; unresolved relationships remain `unknown`.

## Decisions

### Decision: Use `wrapper_integration_test.cu` as the sole baseline builder

The design treats `src/tests/wrapper_integration_test.cu` as the authoritative baseline for this debug change. The staged comparison starts from the values that test loads from CSV or binary assets and from the geometry it overrides from `calc_required_meta.csv`.

Rationale:

- It already exercises the live C++ integration path instead of a separate exporter.
- It already contains the repository's current best-effort remap rules for `subspotData`, `layerLongitudinalCutoffs`, `raySpacing`, `spotDelta`, `sourceDist`, and `refPlaneZ`.
- It narrows the problem to one reproducible fixture-to-wrapper path.

Alternatives considered:

- Compare directly against `dosecal.py` only. Rejected because the immediate bug is in the live wrapper handoff path, and `wrapper_integration_test.cu` is the closest executable baseline.

### Decision: Instrument named checkpoints inside the existing wrapper flow

The design adds named checkpoints to the current wrapper path instead of creating a mirror path. The checkpoints should align with already meaningful stages in `raytracedicom_wrapper.cu` and adjacent code:

- boundary input capture at the test or pybind handoff
- beam basis and sign correction
- source-position, SAD, and `sourceDist` resolution
- ROI and grid normalization
- raw spot lattice reconstruction from `idbeamxy`
- WEQ or BEV import and fan transform setup
- layer energy, peak-depth, and longitudinal-cutoff normalization
- profile-row, beam-parameter-row, and `subspotData` row selection
- exact convolution inputs and per-layer accumulation summaries

Rationale:

- These are the places where the current code can change semantics without changing the public input names.
- A stage map makes it possible to say exactly where alignment is preserved and where it drifts.
- It preserves the current wrapper structure.

Alternatives considered:

- Create a separate "reference wrapper" or replay pipeline. Rejected because it violates the single-path constraint and risks introducing debug drift.

### Decision: Capture both raw and normalized values when decode ambiguity exists

For variables that are currently interpreted multiple ways, the diagnostic output must preserve the raw value and the normalized value or values used by the wrapper. This includes at least:

- `idbeamxy` raw coordinates and the decode formulas used to derive lattice indices or positions
- `sourceDist` explicit values, inferred values, and any fallback to `sad`
- beam-basis vectors before and after sign correction against `beamDirection`
- layer longitudinal cutoff before and after conversion into RTD peak-depth logic
- energy-row selection values such as `energyIdx`, `profileRowIdx`, and remapped `subspotData` rows

Rationale:

- Final values alone are not enough when the suspected bug is a wrong transform or wrong decode formula.
- The Bragg-peak offset may originate from a unit conversion or row-selection drift that only becomes visible if both sides are logged.

Alternatives considered:

- Record only the wrapper's final derived values. Rejected because it would hide the transformation that introduced the mismatch.

### Decision: Classify mismatches by stage and evidence level

The design uses a stage-oriented mismatch classification:

- `aligned`: test input and wrapper state match directly
- `transformed-but-proven`: wrapper changes the representation, but the mapping is explicitly demonstrated
- `mismatched`: wrapper state conflicts with the baseline or with the declared transform rule
- `unknown`: current source and diagnostics do not prove equivalence

The final diagnosis ranks likely causes of dose abnormality and Bragg peak offset using observed mismatch stages rather than unsupported speculation.

Rationale:

- The user asked for concrete mismatch points, not a loose list of suspicions.
- The repository already contains cases where a transform might be legitimate but is currently undocumented.

Alternatives considered:

- Emit only a flat diff log. Rejected because it would not separate proven transforms from real bugs.

### Decision: Keep diagnostics side-effect-free and optional

The diagnostic machinery must not alter the wrapper's algorithmic path. It can collect, summarize, and export state, but it must not create a new computational branch or replace the current kernels.

Rationale:

- The user explicitly requested no behavior change and no parallel implementation.
- A read-only diagnostic path can be validated against the current wrapper output.

Alternatives considered:

- Always-on heavy tracing with behavioral hooks. Rejected because it increases risk of perturbing the path being debugged.

## Risks / Trade-offs

- High-volume trace data -> Mitigation: define named checkpoints with representative summaries and allow optional deeper dumps only when needed.
- Some mappings may still remain unprovable because values are synthesized from `/tables` or inferred indirectly -> Mitigation: classify them as `unknown` instead of inventing equivalence.
- Instrumentation can accidentally perturb execution order or numeric state -> Mitigation: keep capture read-only and verify the same wrapper path remains in use.
- The wrapper contains coupled geometry and energy heuristics, so one visible mismatch may be downstream of another -> Mitigation: preserve stage order and raw-versus-normalized state to identify the first divergence.

## Migration Plan

1. Define the stage model and the data recorded at each checkpoint.
2. Add checkpoint capture to the existing wrapper path and the integration test harness that feeds it.
3. Run the integration fixture and produce a mismatch-localization report.
4. Verify that the diagnostic path preserves the current execution path and numerical behavior.

Rollback strategy: disable or remove the checkpoint plumbing. No data migration is required.

## Open Questions

- What is the best output form for the diagnostic report: structured file, test log, or assertion-oriented diff helper?
- Should the first implementation dump all layers and spots, or summarize representative layers by default and expose full dumps only behind a deeper debug flag?
- What reference metric should be used to connect a confirmed stage mismatch to the observed Bragg peak shift when a full physical truth set is not available in every fixture?
