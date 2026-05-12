## Context

The repository already contains the core pieces of the final-dose chain, but they are not yet closed as one production path:

- `src/tests/wrapper_integration_test.cu` can build a realistic beam, LUT, WEQ, and geometry fixture and call `subsecondWrapper(...)` directly.
- `src/core/raytracedicom_wrapper.cu` now contains the live RTD execution path and recent contract fixes for spot decoding, slice-export LUT alignment, and several CarbonPBS handoff mismatches.
- `src/bindings/raytracedicom_pybind.cpp` exposes CarbonPBS-style compatibility entry points such as `cuFinalDose`, `calcDose`, and `cuCalDose3`, but still carries production-chain gaps around final-dose entry preparation, CT/body synthesis, table-path handling, and public-call verification.
- The user constraint is explicit: keep the wrapper basic form, do not add a parallel implementation, and do not refactor the main flow before shipping the callable production chain.

This change therefore focuses on finishing the existing wrapper-based final-dose path rather than inventing a new path.

## Goals / Non-Goals

**Goals:**

- Make the CarbonPBS-style pybind final-dose entry callable through the existing RTD wrapper path.
- Complete the missing main-chain preparation steps so the pybind entry builds the same effective runtime contract that `wrapper_integration_test.cu` already proves for the direct wrapper path.
- Preserve the wrapper as the single numerical execution path for final-dose production.
- Ensure the public compatibility APIs write final dose with stable output-buffer semantics and stable table lookup behavior.
- Add end-to-end verification that covers both the existing C++ fixture path and the Python-facing pybind entry.

**Non-Goals:**

- Rewriting `subsecondWrapper(...)` into a new architecture.
- Adding a second final-dose engine or a fallback implementation outside the wrapper.
- Reworking dose physics, convolution kernels, or transport math beyond what is strictly required to finish the live chain.
- Redefining the CarbonPBS compatibility API into a different external interface.

## Decisions

### Decision: Keep `subsecondWrapper(...)` as the sole final-dose producer

The completed production chain will continue to marshal inputs into `RTDBeamSettings` and `RTDEnergyStruct`, then call `subsecondWrapper(...)` directly. Compatibility entries such as `cuFinalDose`, `calcDose`, and `cuCalDose3` remain wrappers around that same path.

Rationale:

- The repository already uses `wrapper_integration_test.cu` as the closest executable oracle for the intended path.
- The recent debugging work localized the remaining problems to entry preparation and contract completion, not to the need for a second execution path.
- It satisfies the user's “no parallel implementation” constraint.

Alternatives considered:

- Build a dedicated CarbonPBS final-dose path outside the wrapper. Rejected because it would duplicate logic and immediately reintroduce contract drift.

### Decision: Promote missing fixture-only preparation into shared production helpers

Where the direct wrapper fixture already performs required preparation that the pybind final-dose path does not, the production fix should move only those minimum preparation steps into shared or pybind-side helpers. The prime examples are body/CT placeholder construction, ROI normalization, slice-export LUT handling, and wrapper-compatible geometry defaults.

Rationale:

- `wrapper_integration_test.cu` already encodes the repository's current best-known preparation rules.
- Reusing that knowledge is lower risk than inventing new entry semantics inside the binding layer.
- The goal is contract completion, not a broader refactor.

Alternatives considered:

- Leave fixture preparation and pybind preparation separate. Rejected because the current issue is that the production entry does not yet build the same contract the fixture exercises.

### Decision: Treat `tables_dir` normalization as part of the binding contract

The pybind-facing final-dose path will normalize `tables_dir` before reading RTD reference tables so absolute and relative paths resolve consistently without relying on caller-supplied trailing separators.

Rationale:

- The Python-facing path must be callable as a public interface, not only under one hard-coded directory spelling.
- Path concatenation failures break production entry even when the wrapper path itself is otherwise ready.

Alternatives considered:

- Require callers to always supply the exact slash-terminated path string. Rejected because it leaves a known production footgun in the public API.

### Decision: Preserve CarbonPBS compatibility names but tighten buffer semantics

The compatibility functions will keep their current external names, but the implementation will make output ownership and shape semantics explicit:

- `cuFinalDose` and `calcDose` must write final dose into the provided output grid or returned grid using the existing wrapper path.
- `cuCalDose3` remains compatibility-oriented, but any final-dose behavior it exposes must still come from the same wrapper-backed path rather than a hidden alternate computation.

Rationale:

- Existing Python-side integration expects these entry names.
- The production problem is not API naming; it is incomplete execution-chain behavior underneath those names.

Alternatives considered:

- Remove or rename legacy compatibility functions. Rejected because it is outside the requested scope and would create unnecessary migration work.

### Decision: Verify the completed chain with both C++ and Python entrypoints

The acceptance plan will include:

- the existing `wrapper_integration_test.cu` fixture path as the C++ reference entry
- a pybind-level smoke or integration path that invokes the public final-dose API with realistic fixture inputs

Rationale:

- The final production chain is not complete unless both the underlying wrapper path and the public Python entry can exercise it.
- The recent debugging effort already established the C++ fixture as the best available oracle.

Alternatives considered:

- Verify only the C++ wrapper test. Rejected because the user's target explicitly includes “保证 pybind 可调用”.

## Risks / Trade-offs

- `Fixture logic moved into production glue may accidentally become too broad` → Keep the promoted logic minimal and limited to contract completion that is already proven necessary by the wrapper fixture.
- `Pybind compatibility behavior may still differ from CarbonPBS edge cases not covered by the current fixture` → Use the existing `test_data` case as the primary acceptance baseline and call out remaining uncovered edge cases explicitly.
- `Table lookup and build/export issues can mask wrapper correctness` → Include table-path normalization and module-call verification as first-class acceptance criteria, not as afterthoughts.
- `Finishing the chain may expose deeper algorithm issues after the binding path becomes callable` → Keep the change boundary explicit: this change closes the production path and verifies callability; any remaining numerical discrepancies are localized for follow-up work rather than hidden.

## Migration Plan

1. Identify the exact wrapper-fixture preparation steps that are missing from the pybind final-dose entry.
2. Implement those missing steps in the existing binding or helper path without changing the wrapper's execution shape.
3. Normalize public-entry runtime details such as table lookup and output-buffer handling.
4. Verify the resulting chain through the direct wrapper fixture and the public pybind entry.
5. If issues remain, rollback by reverting the new binding/helper glue while leaving the wrapper core unchanged.

## Open Questions

- Should the completed pybind path synthesize a body-limited CT from ROI indices exactly the way the C++ fixture currently does, or is a lighter-weight wrapper-compatible body placeholder sufficient for all supported final-dose calls?
- What is the minimum Python-side verification that should be required in environments where CUDA runtime availability is inconsistent?
- Are there any remaining public-callers of `cuCalDose3` that depend on undocumented legacy side effects beyond the wrapper-backed final-dose result?
