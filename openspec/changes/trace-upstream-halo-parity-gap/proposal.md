## Why

When `nuclear_correction` is enabled for the current CarbonPBS plan, the final dose gets lower than the `correction=off` run and does not broaden laterally. The evidence is consistent across logs and exported dose bins: `rtd4` sums to `663833.24`, `rtd5_A` sums to `786182.58`, their active bbox is identical (`[88,15,150]..[144,93,206]`), and `y=1..14` stays zero in both, while TPS is much wider (`[78,1,140]..[154,146,216]`).

The new logs also show that halo is not completely absent. `NUC_IDD` and `SUPERP_OUTPUT_NUC_BEV` are being produced, but they are far too small relative to primary IDD. The current wrapper and kernel diverge from `RayTraceDicom-main` in two upstream-facing contracts: halo lattice construction and nuclear IDD indexing. This change isolates those parity gaps from the separate `0..14` transfer clipping problem.

## What Changes

- Audit and realign the halo lattice path against upstream `beam.getSpotIdxToGantry()` semantics when `spotPositionsAreIndices=true`.
- Audit and realign the nuclear IDD memory-step and sentinel-index behavior against upstream `FillIddAndSigmaParams(..., 0, ...)`.
- Add focused diagnostics and regression checks that distinguish:
  - halo generation too weak or built on the wrong lattice
  - downstream transfer clipping that still truncates `0..14`
- Keep the transfer clipping issue explicitly out of scope for this change except where needed to prove that halo generation is or is not upstream-equivalent before transfer.

## Capabilities

### New Capabilities
- `nuclear-halo-upstream-parity`: Align the correction-on halo path with upstream lattice and nuclear-index contracts, and expose diagnostics that prove whether halo generation matches upstream before final dose transfer.

### Modified Capabilities

## Impact

- Affected code:
  - `src/core/raytracedicom_wrapper.cu`
  - `src/algorithms/idd_sigma.cu`
  - wrapper integration and halo audit tooling under `src/tests/` and `tools/`
- Affected runtime behavior:
  - correction-on CarbonPBS final-dose path
  - halo BEV construction and nuclear IDD filling
- Reference inputs and evidence:
  - `output/bg879_beam1035_final_dose-rtd4.bin`
  - `output/bg879_beam1035_final_dose-rtd5_A.bin`
  - `output/bg879_beam1035_final_dose-tps.bin`
  - `RayTraceDicom-main/RayTraceDicom-main/src/kernel_wrapper.cu`
