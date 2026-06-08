## 1. Source Audit

- [ ] 1.1 Trace the live CarbonPBS contracts across `carbonPBS/cudaCalDose.cpp`, `carbonPBS/deviceCalDose.cu`, `src/bindings/raytracedicom_pybind.cpp`, `src/core/raytracedicom_wrapper.cu`, and `src/algorithms/convolution.cu`.
- [ ] 1.2 Record the observed fixture shapes, metadata, and exporter notes from `test_data` and identify the actual `/tables` consumers used by the current compatibility path.

## 2. Contract Dossier

- [ ] 2.1 Build the variable contract table and coordinate-system table from the current source, including units, axes, flattening rules, and current consumers.
- [ ] 2.2 Build the `subspotData` texture layout table and the `dosecal.py` or `test_data` field-to-consumer mapping, including fields that are ignored, collapsed, or test-only.

## 3. Upstream Comparison

- [ ] 3.1 Compare the current repository behavior against `RayTraceDicom-main (1)/RayTraceDicom-main/src/kernel_wrapper.cu`, `gpu_convolution_2d.cu`, and `beam_settings.h`, and enumerate every material deviation in scope.
- [ ] 3.2 Rank the likely causes of Bragg peak position error from the audited contracts and record the rationale for the ranking.

## 4. Audit Gate

- [ ] 4.1 Confirm that this change remains analysis-only and does not modify runtime code before any implementation work begins.
