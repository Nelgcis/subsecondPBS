## ADDED Requirements

### Requirement: Correction-on nuclear flow SHALL add an auditable nuclear dose branch
When `nuclear_correction` is enabled and valid nuclear LUTs are available, the system SHALL execute an auditable nuclear dose branch in addition to reducing the primary branch by `nucWeight`. The branch SHALL expose enough stage data to determine whether nuclear dose is generated, superposed, and transferred to the final dose volume.

#### Scenario: Primary reduction is paired with nuclear generation
- **WHEN** correction-on IDD filling samples a positive `nucWeight` and a positive primary dose increment
- **THEN** the primary result is reduced by `(1 - nucWeight)`
- **AND** a corresponding nuclear result is written for valid mapped nuclear rays using the upstream-equivalent `nucWeight`, `nucRayWeight`, `mass`, and `spotDist` semantics

#### Scenario: Missing halo is attributable to a pipeline stage
- **WHEN** correction-on final dose is lower than correction-off or lacks expected halo broadening
- **THEN** diagnostics identify whether the nuclear contribution disappeared at IDD fill, nuclear superposition, or nuclear transfer

### Requirement: Halo lattice mapping SHALL match the upstream physical PB contract
The correction-on nuclear branch SHALL map nuclear rays from the physical pencil-beam lattice used by upstream `beam.getSpotIdxToGantry()` semantics, not silently from an unrelated dense WEQ/ray lattice when explicit physical spot spacing is available.

#### Scenario: Explicit physical spacing drives halo mapping
- **WHEN** pybind CarbonPBS input provides index-encoded spot positions and explicit per-layer physical spot spacing
- **THEN** the halo mapping uses physical PB spacing and offsets for `nucRayDims`, `nucRayWeights`, `rayToNucSpotIdx`, and `spotDistInRays`

#### Scenario: Dense WEQ fallback is visible
- **WHEN** the system cannot construct a physical PB halo lattice and falls back to a dense WEQ/ray lattice
- **THEN** the fallback is reported explicitly as non-upstream-equivalent for halo lateral profile

### Requirement: Nuclear transfer SHALL be proven separately from final support clipping
The system SHALL verify nuclear BEV-to-dose transfer independently of final support/cropping issues. A correction-on run SHALL report whether `nucTransfDiv` launches over a non-empty dose box and whether it increases final dose from the nuclear BEV texture.

#### Scenario: Non-zero nuclear BEV reaches final dose
- **WHEN** `SUPERP_OUTPUT_NUC_BEV` contains positive finite dose
- **THEN** the nuclear transfer diagnostics report launch bounds and representative texture probes
- **AND** the final dose accumulation after nuclear transfer increases by a positive amount unless the diagnostics identify an empty or out-of-range transfer box

#### Scenario: Final support clipping remains out of scope
- **WHEN** pre-transfer nuclear dose is present and nuclear transfer is proven
- **THEN** any remaining `0..14` final support clipping is treated as a separate transfer-support problem, not as evidence that the nuclear branch failed
