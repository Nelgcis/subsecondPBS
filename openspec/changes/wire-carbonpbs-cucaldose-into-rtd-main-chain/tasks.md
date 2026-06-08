## 1. Contract And Input Alignment

- [x] 1.1 Trace `tests/wrapper_integration_test`, `test_data`, and `src/bindings/raytracedicom_pybind.cpp` to lock the effective CarbonPBS-to-RTD input contract for geometry, grids, ROI, energies, layer counts, spot positions, spot weights, subspot data, and LUT settings.
- [x] 1.2 Confirm or explicitly mark the storage order and unit conventions that can be proven for dose/CT grids, ROI indexing, beam geometry fields, and table-driven depth data before changing the production path.

## 2. Wrapper-Backed Main Path Completion

- [x] 2.1 Fill the missing handoff glue so CarbonPBS-style final-dose inputs reach the existing `raytracedicom_wrapper` main path without introducing a parallel implementation or restructuring the wrapper.
- [x] 2.2 Ensure the completed path uses WEQ-backed transport input semantics and limits any CT-shaped data created at the entry boundary to compatibility-only placeholder use where still required by the wrapper interface.

## 3. Pybind Compatibility And Runtime Wiring

- [x] 3.1 Update the public pybind compatibility entries so `cuFinalDose`, `calcDose`, and any `cuCalDose`-style final-dose entry share the same wrapper-backed preparation, argument normalization, and output dose-grid writeback semantics.
- [x] 3.2 Normalize runtime dependencies required by the public entry, including `tables` path resolution and any compatibility defaults or old logic that must be comment-disabled with a `MODIFIED` label.
- [x] 3.3 Ensure the pybind-facing final-dose path does not silently succeed with all-zero placeholder output under positive-weight inputs, and keep CT placeholder construction limited to compatibility-only fallback behavior.

## 4. Verification

- [x] 4.1 Verify the completed direct wrapper path against `tests/wrapper_integration_test` as the fixture-backed RTD reference.
- [x] 4.2 Verify the public pybind final-dose path with realistic `test_data` inputs and confirm public callability, or document any remaining environment-only blockers explicitly.
- [x] 4.3 Keep `tests/wrapper_integration_test` output semantics aligned with the public pybind dose-grid contract by emitting a directly verifiable pybind-equivalent 3D dose-grid bin artifact.
- [x] 4.4 Add and verify a direct Python-side fixture runner that invokes the public pybind final-dose entry and emits a dose-grid bin artifact for external validation.
