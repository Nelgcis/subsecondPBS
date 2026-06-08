## Context

The repository already contains the core execution path needed for final dose production:

- `tests/wrapper_integration_test` can prepare a realistic beam, grid, WEQ payload, LUT tables, and runtime settings, then call `src/core/raytracedicom_wrapper.cu` directly.
- `src/core/raytracedicom_wrapper.cu` is the live RTD implementation and must remain the single main-path dose producer.
- `src/bindings/raytracedicom_pybind.cpp` exposes CarbonPBS-style compatibility entry points, but the public path still depends on incomplete argument normalization, table resolution, and wrapper-compatible handoff details.
- The user constraint is strict: do not add a parallel implementation, do not refactor the overall architecture, keep `raytracedicom_wrapper` in its current basic form, and only stop using old logic by commenting it with a `MODIFIED` label.

The unresolved work is therefore not a new algorithm. It is the completion of the production contract so that CarbonPBS-style `cuCalDose/cuFinalDose` inputs can enter the same RTD main flow that the wrapper integration fixture already validates. A further constraint is that tracing must stay aligned with `raytracedicom-main`, while any transport quantity currently mediated through CT data in this repository must instead consume WEQ-backed data or a wrapper-compatible placeholder that does not redefine the transport model.

## Goals / Non-Goals

**Goals:**

- Complete the final-dose production main path by routing CarbonPBS-style pybind inputs into the existing wrapper execution path.
- Lock the effective input contract between `test_data`, CarbonPBS-style pybind arguments, and wrapper-consumed RTD structures, including array order, dimensional meaning, ROI indexing, beam geometry, and LUT/layer relationships.
- Ensure tracing-side computation continues to follow `raytracedicom-main` semantics, with WEQ data serving as the transport-driving input wherever this integration path previously depended on CT-derived quantities.
- Keep public pybind compatibility entry points callable and aligned, including `cuFinalDose`, `calcDose`, and any `cuCalDose`-style compatibility wrapper that still surfaces final-dose behavior.
- Verify both the direct wrapper fixture path and the public pybind entry path with repository fixtures.

**Non-Goals:**

- Rewriting `raytracedicom_wrapper` into a new architecture.
- Introducing a second final-dose engine, fallback path, or parallel implementation.
- Performing numerical tuning, physics-model re-derivation, or broad algorithmic changes outside the minimum required contract completion.
- Replacing the external CarbonPBS-style API with a different public interface.

## Decisions

### Decision: Keep `raytracedicom_wrapper` as the sole numerical producer

All production-facing final-dose entries will continue to marshal their inputs into wrapper-compatible RTD structures and execute the existing wrapper path. `tests/wrapper_integration_test` remains the reference oracle for how the completed handoff should behave.

Rationale:

- The repository already has one partially validated main path; duplicating it would reintroduce contract drift.
- The user explicitly disallows a parallel implementation.

Alternatives considered:

- Build a separate CarbonPBS-style dose path outside the wrapper. Rejected because it duplicates core logic and violates the requested boundary.

### Decision: Treat the CarbonPBS-to-wrapper handoff as the main problem to solve

The implementation will focus on completing argument normalization, shape/unit interpretation, WEQ handoff, beam metadata preparation, and table-path resolution rather than altering the dose algorithm itself.

Rationale:

- The direct wrapper fixture already demonstrates that the wrapper can execute once fed the correct contract.
- The current failures are concentrated in production entry preparation and runtime dependency resolution.

Alternatives considered:

- Change wrapper math to match mismatched inputs. Rejected because the issue should first be resolved at the contract boundary.

### Decision: Use WEQ-backed transport semantics and keep CT as compatibility-only input where required

Where the current integrated path still needs a CT buffer to satisfy existing wrapper interfaces, the production entry may synthesize the minimum wrapper-compatible placeholder, but tracing-side transport quantities must be driven by WEQ-aligned data rather than by treating the placeholder CT as authoritative physics input.

Rationale:

- The user requirement explicitly states that all calculations passing through CT data should instead use WEQ data.
- This preserves `raytracedicom-main` tracing semantics while avoiding a larger wrapper refactor in the same change.

Alternatives considered:

- Continue using dummy CT as an implicit transport proxy. Rejected because it obscures the intended WEQ-backed contract and risks further drift from the traced RTD path.

### Decision: Normalize pybind compatibility inputs to match the fixture-backed contract

The pybind layer will be responsible for reconciling CarbonPBS-style input conventions with wrapper expectations, including:

- dose/CT grid dimension, corner, and resolution ordering
- ROI index normalization
- beam geometry fields such as source, beam direction, beam axes, SAD, and reference-plane-related offsets
- layer energy ordering, spot counts, spot positions, subspot data, and LUT indexing
- reference table path normalization for relative and explicit `tables` paths

Rationale:

- The wrapper should stay focused on RTD execution, while input-compatibility fixes belong at the boundary.
- This is the minimum place to make the public API callable without restructuring the core.

Alternatives considered:

- Push all compatibility normalization into the wrapper. Rejected because it would further entangle execution logic with public binding concerns.

### Decision: Keep compatibility entry names and align them on one completed path

`cuFinalDose`, `calcDose`, and any `cuCalDose`-style compatibility entry that still produces final dose will remain externally available, but they must share the same completed wrapper-backed preparation and writeback semantics.

Rationale:

- Existing integrations expect these names.
- Divergent internal paths would create inconsistent behavior and make future debugging harder.

Alternatives considered:

- Leave legacy compatibility functions partially implemented or semantically divergent. Rejected because the change goal is production-chain closure.

## Risks / Trade-offs

- [Argument order or storage conventions remain partially unknown] → The implementation must derive them from `tests/wrapper_integration_test`, `test_data`, and the current wrapper path, and explicitly preserve unknowns rather than inventing semantics.
- [WEQ-backed integration still depends on wrapper CT slots] → Keep any CT placeholder logic minimal, local to the boundary, and clearly documented as compatibility-only rather than transport authority.
- [Public pybind callability can still fail on environment issues such as CUDA runtime mismatch] → Separate build/callability verification from environment-only execution blockers and document any remaining runtime limitation explicitly.
- [Completing compatibility glue may expose deeper numerical discrepancies] → Keep this change scoped to main-path closure and contract alignment; treat any subsequent tuning as follow-up work.

## Migration Plan

1. Trace the existing wrapper fixture contract and identify the minimum preparation missing from the pybind production path.
2. Implement the missing pybind and helper-side normalization needed to feed the existing wrapper path without adding a new execution flow.
3. Preserve or comment-mark old logic with `MODIFIED` labels where necessary, but do not remove fallback code unless proven obsolete.
4. Verify the completed path with both the direct wrapper fixture and the public pybind entry using repository `test_data`.
5. If runtime or numerical issues remain, isolate them as follow-up work after confirming that the production path itself is complete and callable.

## Open Questions

- Which wrapper interface points still require CT-shaped buffers even after WEQ becomes the authoritative transport input, and can those dependencies remain compatibility-only within this change?
- Are any remaining CarbonPBS callers dependent on undocumented `cuCalDose3` side effects beyond final-dose production?
- Which unit conversions and storage orders can be proven directly from current repository fixtures, and which must remain explicitly marked as unresolved until validated during implementation?
