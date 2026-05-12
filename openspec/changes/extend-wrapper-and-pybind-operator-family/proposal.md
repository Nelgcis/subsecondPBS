## Why

The repository now has two related needs that should be solved under one explicit RTD-backed contract:

- `src/tests/wrapper_integration_test.cu` should be able to load `test_data/` directly and compute a `dosegrid` through the live wrapper path without going through pybind.
- the compiled `cudaCalDose1*.so` module should expose the broader CarbonPBS-style function family expected by downstream code, including final-dose, CSC, norm, and biological-map related entries.

The current code does not satisfy that contract cleanly:

- `dosecal.py` only overrides `cuFinalDose` with the local RTD module.
- the current RTD `cuCalDose3` is a fake compatibility entry that zeroes CSC outputs and returns a final-dose grid.
- the wrapper-backed RTD chain can produce final dose, but it does not yet honestly expose the operator-family API surface.

The next change should therefore do two things together:

- make the direct wrapper test the canonical non-pybind final-dose consumer of `test_data`
- make the `.so` surface explicit about which operator-family functions are implemented and which are still staged placeholders

## What Changes

- Lock one shared CarbonPBS-style input contract between `test_data`, `wrapper_integration_test`, pybind entry points, and the RTD wrapper.
- Keep final-dose computation on the existing RTD wrapper-backed path for both direct-wrapper and pybind-facing final-dose calls.
- Extend the `.so` API surface so the expected CarbonPBS-style function family is present.
- Replace fake CSC-style compatibility behavior with honest semantics: real implementation where available, explicit unsupported behavior where not yet implemented.
- Stage the future RTD-native spot-preserving operator work needed for real CSC and row-norm outputs.

## Capabilities

### New Capabilities

- `wrapper-and-pybind-operator-family`: Use one RTD-backed contract for direct-wrapper final dose and the broader CarbonPBS-style pybind operator family.

### Modified Capabilities

- `wrapper-final-dose-production-chain`: The final-dose chain remains wrapper-backed, but now lives alongside an explicitly staged operator-family surface.

## Impact

- Affected code: `src/tests/wrapper_integration_test.cu`, `src/bindings/raytracedicom_pybind.cpp`, `src/core/raytracedicom_wrapper.cu`, shared RTD integration headers, and related test-data loaders.
- Affected APIs: `cuCalDose`, `cuFinalDose`, `cuCalDoseNorm`, `cuCalDose3`, `cuFinalDoseAndRBEMap`, and any explicitly exposed biological helper entries.
- Affected runtime surface: direct wrapper fixture execution, pybind export naming, and CarbonPBS-style input normalization.
