## Why

The repository now has a traced and partially corrected `wrapper_integration_test.cu -> raytracedicom_wrapper.cu` handoff, but the final production chain is still not closed end to end for CarbonPBS-style final-dose calls. The missing pieces are now concentrated in the live wrapper and pybind path, so the next change should finish that path directly instead of introducing a second implementation.

## What Changes

- Complete the existing `wrapper_integration_test` and `raytracedicom_wrapper` main chain so it can produce final dose through the live RTD wrapper path without restructuring the wrapper.
- Fill only the missing production-link pieces around the current chain: pybind argument normalization, final-dose entry preparation, table-path resolution, output writeback semantics, and any wrapper-side glue still required for the live path to run as a production entry.
- Keep `cuFinalDose`, `calcDose`, and the direct wrapper API on one execution path that routes through the existing RTD wrapper instead of a mirrored or fallback implementation.
- Add focused end-to-end verification using the existing test fixture and pybind entry so the completed chain is callable and produces final-dose outputs through the intended public interface.
- Preserve the wrapper's current basic structure and avoid physics refactors, parallel pipelines, or algorithm rewrites while finishing the production chain.

## Capabilities

### New Capabilities
- `wrapper-final-dose-production-chain`: Complete the existing wrapper-based final-dose production path so CarbonPBS-style pybind callers can invoke the live RTD wrapper and receive final-dose output through the supported public API.

### Modified Capabilities
None.

## Impact

- Affected code and integration surfaces: `src/core/raytracedicom_wrapper.cu`, `src/bindings/raytracedicom_pybind.cpp`, `src/tests/wrapper_integration_test.cu`, `test_data`, `CMakeLists.txt`, and any helper code required to keep the final-dose pybind module callable.
- Affected APIs: `cuFinalDose`, `calcDose`, `cuCalDose3`, and the direct `raytracedicom_wrapper(...)` Python entry.
- Affected runtime dependencies: RTD reference table lookup under `/tables`, Python module build/export wiring, and CarbonPBS-style array shape or ordering contracts passed into the wrapper.
- Runtime constraint: the change must preserve the current wrapper-centric execution path and must not add a parallel final-dose implementation.
