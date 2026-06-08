## 1. Baseline and Stage Model

- [ ] 1.1 Map every baseline input built by `src/tests/wrapper_integration_test.cu` to the corresponding wrapper entry field, including geometry overrides from `calc_required_meta.csv`.
- [ ] 1.2 Define the named debug checkpoints and the per-stage fields that must be captured for geometry, grids, ROI, spot decoding, WEQ or BEV transfer, energy mapping, LUT mapping, and convolution setup.

## 2. Wrapper Instrumentation

- [ ] 2.1 Add read-only checkpoint capture to the existing `src/core/raytracedicom_wrapper.cu` path for beam-basis construction, `source` or `sad` or `sourceDist` handling, and ROI or grid normalization.
- [ ] 2.2 Add read-only checkpoint capture for `idbeamxy` decoding, raw spot lattice reconstruction, WEQ-to-BEV transfer, layer energy or cutoff normalization, `profileRowIdx` or `beamParaData` selection, `subspotData` row or channel mapping, and exact RTD convolution inputs.

## 3. Mismatch Localization

- [ ] 3.1 Extend the wrapper integration workflow so it can compare test-built baseline values against wrapper checkpoint values and classify each stage as aligned, transformed-but-proven, mismatched, or `unknown`.
- [ ] 3.2 Produce a stage-ordered mismatch report that highlights the first confirmed divergence points and ranks the most likely causes of abnormal dose and Bragg peak displacement.

## 4. Verification

- [ ] 4.1 Verify that the diagnostic path preserves the wrapper's single execution path and does not introduce a parallel implementation or main-flow refactor.
- [ ] 4.2 Run the wrapper integration workflow with diagnostics and confirm that unresolved relationships are reported explicitly as `unknown` rather than silently inferred.
