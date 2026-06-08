## Why

The repository already has contract-audit work, but the current dose anomaly and Bragg peak offset problem is now in the live handoff path between `src/tests/wrapper_integration_test.cu` and `src/core/raytracedicom_wrapper.cu`. Before changing any physics or rewriting any integration layer, the project needs one debug-oriented change that traces the existing wrapper execution stage by stage and identifies the exact variable, coordinate, grid, energy, spot, subspot, or LUT mismatch points.

## What Changes

- Create a wrapper-integration alignment debug change that treats `src/tests/wrapper_integration_test.cu` as the reference input builder and comparison baseline for the current `subsecondWrapper(...)` path.
- Add staged traceability requirements for the existing wrapper flow, covering beam-basis construction, source or SAD inference, ROI and grid normalization, `idbeamxy` decoding, raw spot lattice reconstruction, WEQ-to-BEV transfer, layer energy and longitudinal-cutoff mapping, profile or beam-parameter row selection, `subspotData` row or channel mapping, and exact RTD convolution inputs.
- Require a mismatch-localization report that compares test-provided inputs to wrapper-normalized intermediates and classifies each checked stage as aligned, transformed-but-proven, mismatched, or `unknown`.
- Keep the wrapper as one execution path: no parallel implementation, no alternate wrapper, no main-flow refactor, and no intended algorithm-behavior change in this change.

## Capabilities

### New Capabilities
- `wrapper-integration-alignment-debug`: Trace and localize integration mismatches between `wrapper_integration_test.cu` inputs and the live `raytracedicom_wrapper.cu` execution path.

### Modified Capabilities
None.

## Impact

- Affected code and diagnostics surfaces: `src/tests/wrapper_integration_test.cu`, `src/core/raytracedicom_wrapper.cu`, `src/algorithms/convolution.cu`, `src/bindings/raytracedicom_pybind.cpp`, and any helper structures used to record staged debug state.
- Affected debug scope: beam geometry, grid and ROI metadata, WEQ and BEV transforms, energy-layer mapping, profile or beam-parameter lookup, `subspotData` mapping, and exact convolution inputs.
- Runtime constraint: this change must preserve the current wrapper structure and single execution path while exposing enough stage data to pinpoint the mismatch responsible for abnormal dose and Bragg peak displacement.
