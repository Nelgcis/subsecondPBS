## 1. Reproduce and lock the evidence

- [ ] 1.1 Archive the representative correction-on/off evidence for `bg879_beam1035`, including totals, active bbox, and the unchanged `y=1..14` zero band from `rtd4`, `rtd5_A`, and TPS.
- [ ] 1.2 Capture the current correction-on wrapper audit lines that prove halo is running but too small, including representative `IDD`, `NUC_IDD`, `RSIGMA`, `NUC_RSIGMA`, and `SUPERP_OUTPUT_NUC_BEV` summaries.
- [ ] 1.3 Add or update a lightweight audit script or test helper that reports halo-grid spacing, physical PB spacing, and valid `rayToNucSpotIdx` occupancy for one representative layer.

## 2. Restore halo lattice parity with upstream

- [ ] 2.1 Update the correction-on wrapper path so index-encoded CarbonPBS beams with explicit physical spacing build halo mapping from the physical PB lattice contract instead of defaulting to the dense WEQ lattice.
- [ ] 2.2 Verify that `buildHaloLatticePlan(...)` receives physical PB `spotDelta/spotOffset` semantics and that the audit no longer reports `haloGridDelta=(1,1)` for the current `4.3/3.1` mm plan.
- [ ] 2.3 Add a regression check that fails if correction-on halo silently falls back to the dense WEQ lattice for the current pybind input contract.

## 3. Restore nuclear IDD indexing parity with upstream

- [ ] 3.1 Compare the current `nucMemStep` and `nucIdx` path against upstream `FillIddAndSigmaParams(..., 0, ...)` and choose the implementation that preserves upstream mapped/unmapped semantics.
- [ ] 3.2 Implement the selected parity fix in the wrapper or kernel, with an explicit regression that proves invalid `nucIdcs` do not become valid after depth-step shifting.
- [ ] 3.3 Add a focused audit or test that demonstrates correct indexing for both unmapped rays and mapped halo rays across multiple depth steps.

## 4. Verify halo parity separately from transfer clipping

- [ ] 4.1 Re-run the representative plan with correction on/off and confirm that pre-transfer halo diagnostics show materially stronger nuclear output once parity fixes land.
- [ ] 4.2 Compare correction-on vs correction-off final dose totals and bbox again, and document whether any remaining lack of broadening is now attributable to the existing `0..14` transfer clipping.
- [ ] 4.3 If pre-transfer halo parity is restored but final dose remains clipped, open or link a separate change dedicated to final transfer support expansion rather than continuing to tune halo physics here.
