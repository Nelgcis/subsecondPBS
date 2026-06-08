## Why

The current repository exposes one CarbonPBS integration chain but does not expose one stable input contract. `carbonPBS/*`, `test_data`, `tps_py/dosecal.py`, the pybind compatibility layer, and the RayTraceDicom wrapper all interpret some inputs differently, so `calDoseSubsecond` cannot be implemented safely until the variable meanings, shapes, coordinate frames, and buffer-order rules are locked explicitly.

## What Changes

- Create an algorithm-neutral CarbonPBS to test-data to RayTraceDicom input-contract change for the current repository.
- Produce a source-anchored contract dossier that locks variable meanings, shapes, axes, units, coordinate frames, and normalization rules across `carbonPBS`, `test_data`, the pybind compatibility layer, and the RTD wrapper.
- Define the canonical `subspotData` texture contract, canonical ROI representation, beam-geometry mapping, and `doseGrid` or `ctGrid` axis-order rules that future `calDoseSubsecond` work must follow.
- Cross-check `dosecal.py` or `test_data` exports against the actual legacy CarbonPBS consumers and the actual RTD-wrapper consumers, then classify each field as consistent, ambiguous, shape-risk, unit-risk, order-risk, or `unknown`.
- Deliver a high-risk mismatch list and a future-input checklist without changing numerical dose behavior.

## Capabilities

### New Capabilities
- `carbonpbs-rtd-input-contract`: Lock the canonical CarbonPBS to RayTraceDicom input contract and the associated normalization or validation checklist for future `calDoseSubsecond` implementation.

### Modified Capabilities
None.

## Impact

- Affected code and integration surfaces: `carbonPBS/cudaCalDose.cpp`, `carbonPBS/deviceCalDose.cu`, `src/bindings/raytracedicom_pybind.cpp`, `src/core/raytracedicom_wrapper.cu`, `src/algorithms/convolution.cu`, `test_data`, `tps_py/dosecal.py`, and `/tables`.
- Affected APIs: `calcDose`, `cuFinalDose`, `cuCalDose3`, the direct `raytracedicom_wrapper(...)` Python API, and any future `calDoseSubsecond` input builder or validator.
- Runtime constraint: this change is contract-locking only. It does not change algorithm math, kernel selection, or physical dose behavior.
