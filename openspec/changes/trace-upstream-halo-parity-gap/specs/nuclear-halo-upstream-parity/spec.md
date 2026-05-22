## ADDED Requirements

### Requirement: Correction-on halo mapping SHALL use the physical PB lattice contract
When `nuclear_correction` is enabled for a beam with `spotPositionsAreIndices=true` and explicit physical spot spacing, the halo branch SHALL use a physical PB lattice expressed in gantry-space millimeters. It SHALL NOT silently use the dense WEQ/ray lattice as the effective halo grid for those inputs.

#### Scenario: Explicit CarbonPBS spacing drives the halo grid
- **WHEN** a CarbonPBS beam provides index-encoded spot positions together with explicit `spotDelta` or `layerSpotDeltas`
- **THEN** the halo plan uses physical PB `spotDelta` and `spotOffset` semantics rather than the dense WEQ header lattice

### Requirement: Nuclear IDD indexing SHALL preserve upstream sentinel semantics
The nuclear IDD path SHALL preserve invalid mapping sentinels across all depth steps. A ray whose nuclear mapping is invalid SHALL not write nuclear IDD or nuclear sigma values at any step. A ray whose nuclear mapping is valid SHALL advance through nuclear memory using the selected parity contract without changing the meaning of invalid mappings.

#### Scenario: Unmapped rays remain unmapped for all steps
- **WHEN** `nucIdcs[idx2d]` is invalid for a primary ray and `firstStep > 0`
- **THEN** the kernel performs no nuclear write for that ray at any depth step

#### Scenario: Mapped rays advance with a consistent step contract
- **WHEN** `nucIdcs[idx2d]` is valid for a primary ray
- **THEN** successive nuclear writes advance according to the selected upstream-equivalent memory-step rule without corrupting neighboring halo rays

### Requirement: Halo parity diagnostics SHALL distinguish halo-generation failure from transfer clipping
When correction-on halo diagnostics are enabled, the system SHALL report enough information to determine whether missing broadening originates before or after final transfer. The report SHALL include halo-grid spacing, physical PB spacing, spot-distance-in-rays, nuclear memory-step semantics, and representative primary-vs-nuclear BEV dose summaries before transfer.

#### Scenario: Audit output proves which contract was used
- **WHEN** a correction-on plan runs with halo audit enabled
- **THEN** the emitted diagnostics identify whether the halo branch used physical PB spacing or WEQ fallback, and whether pre-transfer nuclear dose is materially present before final transfer
