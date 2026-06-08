## Why

The repository currently contains three different truths about the CarbonPBS interface: the original `carbonPBS/*` kernels, the current pybind compatibility adapter, and the heavily modified RTD wrapper path. Before any implementation, the project needs one source-anchored audit that states what each variable actually means, how coordinates and textures are interpreted, where the current code deviates from `RayTracedicom_main`, and which mismatches are most likely to move the Bragg peak.

## What Changes

- Replace the existing implementation proposal with an audit-only change for the CarbonPBS compatibility path.
- Produce a source-anchored audit dossier covering `carbonPBS/cudaCalDose.cpp`, `carbonPBS/deviceCalDose.cu`, `src/core/raytracedicom_wrapper.cu`, `src/algorithms/convolution.cu`, `src/bindings/raytracedicom_pybind.cpp`, `test_data`, and `/tables`.
- Include the six required deliverables in the dossier: a variable contract table, a coordinate-system table, a `subspotData` texture layout table, a `test_data/dosecal.py` field-to-consumer mapping, a complete deviation list versus `RayTracedicom_main`, and a ranked list of likely Bragg peak position error causes.
- Keep the change analysis-only: no runtime code changes, no API rewrites, and no numerical behavior changes in this change set.

## Capabilities

### New Capabilities
- `carbonpbs-contract-audit`: Document the current CarbonPBS-to-RTD contract, upstream drift, and Bragg-peak risk points before implementation begins.

### Modified Capabilities
None.

## Impact

- Affected artifacts: `openspec/changes/carbonpbs-contract-audit/{proposal,design,tasks}.md` and `openspec/changes/carbonpbs-contract-audit/specs/carbonpbs-contract-audit/spec.md`.
- Audited runtime surfaces: `carbonPBS/*`, `src/bindings/raytracedicom_pybind.cpp`, `src/core/raytracedicom_wrapper.cu`, `src/algorithms/convolution.cu`, `test_data`, and `/tables`.
- Upstream comparison basis: `RayTraceDicom-main (1)/RayTraceDicom-main/src/{kernel_wrapper.cu,gpu_convolution_2d.cu,beam_settings.h}`.
- Runtime impact in this change: none. This change is documentation and audit only.
