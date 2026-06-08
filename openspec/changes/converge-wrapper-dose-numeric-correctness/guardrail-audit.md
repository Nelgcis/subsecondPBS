# Guardrail Audit Snapshot

Date: 2026-05-07

Scope:
- `src/core/raytracedicom_wrapper.cu`
- `src/bindings/raytracedicom_pybind.cpp`

Purpose:
- classify the current fallback / silent-skip / heuristic branches before more kernel-side cleanup lands
- separate branches that are already acceptable from branches that need explicit metadata and branches that should become hard runtime errors

## Prototype-Equivalent And Allowed

- `buildRawSpotGrid(...)` and `buildHaloLatticePlan(...)`: zero-weight spots are treated as valid no-dose inputs and skipped, while positive-weight spots that cannot be decoded or rasterized fail the beam. This matches the current change decision that zero-particle CarbonPBS spots are valid plan data rather than malformed geometry.
- `bind_output_dose_array(...)` plus `commit_output_dose(...)`: explicit conversion between wrapper-native `(Z,Y,X)` payload order and Python-facing `(Nx,Ny,Nz)` C-order is acceptable as long as the layout contract stays explicit. The new `tools/compare_final_dose_bins.py` audit script exists specifically to keep this conversion from being mistaken for a physical dose difference.

## Requires Explicit Exported Metadata Or A Proven Contract

- `parse_slice_mode_env(...)`, `select_slice_energies(...)`, and the default `tail` slice-mode assumption for IDD/profile tables: this is still an export-contract heuristic, not a proven CarbonPBS semantics lock.
- `profileSetting[1] *= logical_depth_n / profile_depth_n` in the pybind loader: stretching the depth step for compact profile exports requires explicit exporter metadata that proves uniform-decimation semantics.
- `derive_layer_longitudinal_cutoffs(...)`: collapsing per-spot `longitudalCutoff` down to one value per layer is only acceptable when the exporter proves within-layer consistency.
- `to_roi_linear_indices(...)`: the current 1D input path still guesses whether a length-multiple-of-3 vector is XYZ triplets or already-linear indices. That ambiguity needs explicit metadata or a stricter entry contract.
- `autoDetectCtType(...)`: CT/HU/HU+1000/density inference is still heuristic. If the RTD compatibility path depends on CT interpretation rather than purely on exported WEQ, the caller must eventually provide explicit semantics.
- `normalizeLongitudinalCutoffToMm(...)` and the `energyDepthToMm` auto-detection path: current unit adaptation still contains legacy heuristic behavior.
- `profileRows` / `profileChannels` last-resort inference from tensor size in `raytracedicom_wrapper.cu`: acceptable only as an audit aid until the exporter proves exact tensor layout semantics.
- `sourceDist` / SAD recovery from spot directions (`estimateVirtualSourceDistancesFromSpots(...)`, `sad_cm` fallback, per-axis source distance inference): still needs an explicit virtual-source-distance contract or a consistency proof.

## Must Become Hard Runtime Error Or Explicit Unsupported-Input Behavior

- Legacy CPB fallback path in `raytracedicom_wrapper.cu` when `hasRawSpotLattice == false`: this is the main degraded-output path and must not remain silently available unless it is proven numerically equivalent.
- `beamSettings.sourceDist = (0,0)` fallback in the dictionary-style wrapper entry and later `sad/sourceDist` recovery: missing exported source-distance geometry must not silently collapse to scalar SAD.
- `cutoff_obj` in `cu_final_dose_py(...)`: the final-dose binding still accepts the CarbonPBS `cutoff` argument and discards it. That must stay explicitly unsupported or become a hard error when the caller expects it to alter physics.
- Carbon-specific profile/subspot/overall-weight behavior currently disabled by hardcoded `useCarbonSubspotConvolution = false`, `useCarbonProfileSigmaOverride = false`, and `useCarbonProfileOverallWeight = false`: those inputs should not remain silently ignored.
- Out-of-range ROI linear indices in `to_roi_linear_indices(...)` are still only warning-logged. Once the parity harness is locked, invalid ROI indices should fail immediately.
- Out-of-range energy clamp in the per-layer LUT lookup currently warns but still clamps to the LUT edge. For correctness work, that should eventually fail unless an explicit out-of-range policy is defined.

## Already Hard-Fail And Keep

- `validateWrapperInputs(...)`, `validateBeamSettings(...)`, and the positive-weight rasterization checks in raw-spot reconstruction already fail on malformed beam-critical contracts and should stay fail-fast.
- `nuclear_correction=true` without a raw spot lattice already throws instead of silently falling back to the non-halo CPB path.
- WEQ payload size checks against the 9-value header already hard-fail when the header-declared used prefix is inconsistent with the uploaded payload.

## Notes For Ongoing Work

- The audit above is intentionally narrower than a full stage-by-stage numeric diagnosis. It is only the contract/fallback classification needed by tasks `0.05` and `0.06`.
- The current A/C slice-sum contradiction question is not a physical geometry issue. For the same canonical `(x,y,z)` array, per-axis slice-sum reductions must agree on the sign of the total difference. If they do not, the comparison mixed raw layout semantics or reshape order.
