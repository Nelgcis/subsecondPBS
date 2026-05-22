## 1. Lock the Upstream Contract

- [ ] 1.1 Document the exact upstream correction-on sequence from `kernel_wrapper.cu`: LUT upload, `devNucRayWeights`, `devNucSpotIdx`, `fillIddAndSigma`, nuclear superposition, and `nucTransfDiv`.
- [ ] 1.2 Record the upstream formulas for `res`, `nucRes`, `nucRSigmaEff`, and `spotDistInRays`.
- [ ] 1.3 Confirm whether each local stage uses the same units as upstream: mm fan geometry, physical PB spacing, WEPL depth axis, and texture depth coordinates.

## 2. Prove Whether Nuclear IDD Is Generated

- [x] 2.1 Run a representative correction-on plan with existing audits enabled and capture `IDD`, `NUC_IDD`, `RSIGMA`, and `NUC_RSIGMA` summaries for first/mid/last representative layers.
- [x] 2.2 Add or identify a diagnostic for valid `rayToNucSpotIdx` count, invalid count, and mapped-ray overlap with non-zero primary ray weights.
- [x] 2.3 Compare expected nuclear magnitude from `nucWeightRange`, `sum_nucRayWeight`, `sum_rayWeight`, and `spotDistInRays^2` against actual `NUC_IDD`.
- [x] 2.4 If `NUC_IDD` is near zero, isolate whether the cause is missing mapped rays, zero `nucRayWeights`, tiny `nucWeight`, invalid `spotDist`, or mass/depth sampling.

## 3. Prove Whether Nuclear Superposition Is Active

- [x] 3.1 Capture nuclear tile radius counters or summaries for representative layers.
- [x] 3.2 Confirm `NUC_RSIGMA` is finite wherever `NUC_IDD` is positive.
- [x] 3.3 Confirm `SUPERP_OUTPUT_NUC_BEV` has non-zero sum/nnz and shows a broader BEV footprint than primary at comparable depths.
- [x] 3.4 If `NUC_IDD` is non-zero but `SUPERP_OUTPUT_NUC_BEV` is zero or not broader, compare local `performNuclearTileBasedSuperposition` inputs against upstream `kernelSuperposition` launch inputs.

## 4. Prove Whether Nuclear Transfer Adds Dose

- [x] 4.1 Capture `nucTransfDiv` launch bounds, grid size, halo offset, halo delta, and `willLaunch`.
- [x] 4.2 Add or identify probes that map representative dose voxels into nuclear BEV texture coordinates and report in/out status.
- [x] 4.3 Measure final dose sum immediately before and after `nucTransfDiv` for representative layers.
- [x] 4.4 If `SUPERP_OUTPUT_NUC_BEV` is non-zero but final dose does not increase, compare local texture z-slab and `invertAndShift(..., -beamFirstInside)` semantics with upstream.

## 5. Decide the Root Cause and Next Change

- [x] 5.1 Classify the failure as one of: lattice/mapping, IDD/LUT/magnitude, nuclear superposition, nuclear transfer, or a combination.
- [x] 5.2 Write the implementation plan only after the failing stage is proven.
- [x] 5.3 Keep any remaining `0..14` final support issue out of this change; if needed, open a separate transfer-support change after nuclear dose is proven to exist pre-transfer.

## Session Notes

- audit4 proved the nuclear branch is generated and transferred: `NUC_IDD`, `NUC_RSIGMA`, `SUPERP_OUTPUT_NUC_BEV`, and `NUC_DELTA` are all non-zero for representative layers.
- The failing stage is lattice/mapping: explicit physical PB spacing existed, but halo plans still used the dense WEQ/ray lattice, so `haloGridDelta=(1,1)` while `physicalPBDelta` was 4.3/3.7/3.1 mm depending on layer.
- Superseded implementation plan: directly requiring decoded TPS/dosecal spots to lie on an explicit physical PB lattice is invalid. dosecal spots can be irregular before RayTraceDicom convolution, so explicit `spotSpacingX/Z` must drive halo scale and normalization, not raw-spot alignment.
- The `0..14` support/cropping issue remains intentionally out of scope for this change.

## 6. Revised Halo Lattice Implementation for Irregular TPS Spots

- [x] 6.1 Remove the hard alignment requirement that decoded raw spots must be integer multiples of `layerSpotDeltas`; keep a diagnostic for fractional offsets instead of failing.
- [x] 6.2 Implement option A: for each layer, build a virtual physical halo lattice with `haloGridDelta=layerSpotDeltas[layer]`, choose a bounded physical origin from decoded spot bounds, and rasterize irregular decoded spot weights onto that lattice with weight conservation.
- [x] 6.3 Prefer bilinear/area-conserving rasterization over nearest-cell assignment so total weight and first lateral moment are less distorted when TPS spots are off-lattice.
- [x] 6.4 Keep upstream center-to-center `rayToNucSpotIdx` semantics by mapping virtual halo cell centers to the nearest primary ray centers, while computing `spotDistInRays` from the explicit per-layer physical spacing.
- [x] 6.5 Add memory guards and audits: report decoded spot bounds, virtual halo origin/dims, padded dims, weight in/out, mapped center count, and fail before allocation if the virtual halo lattice is unexpectedly larger than the primary ray lattice or exceeds a fixed sanity cap.
- [ ] 6.6 Verify correction-on representative layers show `source=explicit_physical_pb_rasterized_lattice`, `haloGridDelta≈physicalPBDelta`, conserved `sum_nucRayWeight/sum_rawSpotWeight`, non-zero `NUC_IDD`, and positive `NUC_DELTA`.
