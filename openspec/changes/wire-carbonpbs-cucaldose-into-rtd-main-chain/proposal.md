## Why

`tests/wrapper_integration_test` already proves that the repository can assemble a valid `RTDBeamSettings` and `RTDEnergyStruct` contract and run `src/core/raytracedicom_wrapper.cu`, but the production-facing CarbonPBS-style pybind path is still not fully connected. The remaining gap is no longer a new algorithm; it is the missing contract completion between `test_data` style inputs, `cuCalDose/cuFinalDose` style arguments, and the live RayTraceDicom wrapper path, especially where tracing must follow `raytracedicom-main` semantics while CT-mediated transport is replaced by WEQ data.

## What Changes

- Complete the final-dose production main path by routing CarbonPBS-style pybind inputs through the existing `raytracedicom_wrapper` flow without adding a parallel implementation.
- Define and lock the production input contract between `test_data`, CarbonPBS-style `cuCalDose/cuFinalDose` arguments, pybind marshalling, and wrapper-consumed RTD structures, including shape, ordering, units, ROI format, beam geometry, and layer/LUT indexing.
- Promote only the missing handoff glue needed for the live path to execute end to end: WEQ-backed tracing inputs, wrapper-compatible geometry defaults, table lookup normalization, and output dose-grid writeback semantics.
- Keep `raytracedicom_wrapper` in its current basic form, keep tracing behavior aligned with `raytracedicom-main`, and only disable old logic by commenting it with a `MODIFIED` label when strictly necessary.
- Add verification that covers both the direct wrapper fixture path and the public pybind `cuCalDose/cuFinalDose` path with realistic `test_data` inputs.

## Capabilities

### New Capabilities
- `carbonpbs-cucaldose-rtd-main-chain`: Complete the wrapper-backed production path that maps CarbonPBS-style final-dose inputs into the existing RayTraceDicom main flow and returns a final dose grid through the supported pybind API.

### Modified Capabilities

## Impact

- Affected code: `src/core/raytracedicom_wrapper.cu`, `src/bindings/raytracedicom_pybind.cpp`, `src/tests/wrapper_integration_test.cu`, related helper code under `src/utils`, and targeted `test_data` fixtures used for public-entry verification.
- Affected APIs: `cuCalDose`-style pybind argument contract, `cuFinalDose`, `calcDose`, `cuCalDose3`, and the direct wrapper-facing Python API.
- Affected runtime dependencies: RTD reference table lookup under `/tables`, WEQ payload preparation, ROI/grid shape conventions, and Python-to-C++ array layout assumptions.
- Constraint: no parallel engine, no whole-flow rewrite, and no algorithmic simplification of the existing RTD tracing or dose pipeline.
