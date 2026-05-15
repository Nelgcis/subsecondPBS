## Why

`tests/wrapper_integration_test` and the current pybind path now reach the live `raytracedicom_wrapper` main chain, but the produced dose is not yet numerically trustworthy. The remaining problem is stage-by-stage numerical convergence: Bragg peak position error, abnormal dose shape, energy-layer misalignment, and possible depth-direction writeback mistakes must now be isolated and corrected without changing the wrapper's overall form.

## What Changes

- Use `tests/wrapper_integration_test` as the primary executable debugging baseline for the already-connected main path and drive numerical convergence against repository fixture data.
- Audit and correct the main chain stage by stage, specifically covering CPB geometry, reference-plane geometry, WEPL generation, LUT lookup, sigma transport, IDD/CIDD mapping, superposition, and final dose-grid writeback.
- Add explicit per-stage diagnostics, comparisons, and correction checkpoints so each stage can report its likely error source and the effect of the applied fix.
- Fix confirmed causes of Bragg peak misplacement, abnormal depth-dose behavior, energy layer mismatches, and depth-direction writeback errors while keeping `raytracedicom_wrapper` in its current basic structure.
- Convert current non-equivalent fallback and heuristic branches into explicit contracts: legacy CPB fallback, ambiguous unit inference, sliced LUT reconstruction, and missing virtual-source-distance semantics must either be proven equivalent to the RTD prototype or fail loudly.
- Keep pybind and `wrapper_integration_test` numerically aligned on the same wrapper-backed path so Python-facing dose output reflects the same corrected chain.

## Capabilities

### New Capabilities
- `wrapper-dose-numeric-correctness`: Converge the already-connected RTD wrapper main path to numerically correct final-dose behavior by stage-wise debugging, correction, and error-source reporting against fixture-backed reference data.

### Modified Capabilities

## Impact

- Affected code: `src/core/raytracedicom_wrapper.cu`, `src/tests/wrapper_integration_test.cu`, `src/bindings/raytracedicom_pybind.cpp`, and targeted helpers used by CPB, WEPL, LUT, sigma, IDD/CIDD, and dose writeback handling.
- Affected verification surfaces: `tests/wrapper_integration_test`, Python-facing pybind final-dose calls, and fixture-driven bin outputs under `test_data` and `output/`.
- Affected export/input contracts: sliced `idd/profile` fixture semantics, per-layer/per-spot cutoff semantics, and virtual source distance metadata now need explicit auditing because they currently drive plan-dependent wrapper behavior.
- Affected APIs: public pybind final-dose entries remain the same externally, but their numerical behavior is expected to converge toward the wrapper fixture and reference dose outputs.
- Constraint: no large-scale refactor, no parallel implementation, no architecture rewrite, and no scope expansion into unrelated performance work.
