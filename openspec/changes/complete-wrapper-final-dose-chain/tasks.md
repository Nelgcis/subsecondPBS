## 1. Entry Contract Completion

- [ ] 1.1 Promote the minimum wrapper-fixture preparation required for final-dose production into the live pybind entry, including ROI normalization, CT/body placeholder preparation, and wrapper-compatible beam or LUT handoff state.
- [ ] 1.2 Normalize public-entry runtime inputs such as `tables_dir`, output-buffer shape/order handling, and compatibility defaults so the final-dose path is callable through the supported pybind API.

## 2. Wrapper-Backed Final-Dose Path

- [ ] 2.1 Update `cuFinalDose` and `calcDose` so they complete the missing main-chain glue and execute final dose through the existing `subsecondWrapper(...)` path without adding a parallel implementation.
- [ ] 2.2 Update any wrapper-backed compatibility entry that still exposes final-dose behavior, including `cuCalDose3`, so public final-dose semantics remain aligned with the same completed wrapper path.

## 3. Build And Packaging

- [ ] 3.1 Ensure the pybind module build/export wiring produces a callable public module with the intended final-dose entry names.
- [ ] 3.2 Ensure RTD reference table lookup and other runtime dependencies resolve correctly from the completed pybind final-dose path in both relative and explicit-path usage.

## 4. Verification

- [ ] 4.1 Verify the completed chain with `src/tests/wrapper_integration_test.cu` using the existing fixture as the direct wrapper-path reference.
- [ ] 4.2 Verify the public pybind final-dose API with realistic fixture inputs from `test_data` and confirm output writeback semantics or document any remaining environment-only blockers explicitly.
