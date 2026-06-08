## ADDED Requirements

### Requirement: The wrapper-backed main path SHALL support stage-wise numerical convergence against the fixture baseline
The system SHALL use `tests/wrapper_integration_test` as the primary debugging and validation baseline for numerical convergence of the already-connected wrapper-backed dose path. Numerical investigation SHALL proceed by explicit pipeline stage rather than by broad untargeted tuning.

#### Scenario: Wrapper fixture remains the numerical baseline
- **WHEN** numerical correctness work is performed on the connected main path
- **THEN** `tests/wrapper_integration_test` is used as the primary executable baseline before confirming results through pybind

#### Scenario: Numerical debugging follows explicit stages
- **WHEN** a dose discrepancy is investigated
- **THEN** the work identifies the affected stage among CPB geometry, reference-plane geometry, WEPL, LUT, sigma transport, IDD/CIDD, superposition, or dose writeback instead of applying undifferentiated global tuning

### Requirement: Each major numerical stage SHALL produce an error-source and correction summary
The system SHALL provide a stage-wise summary for each corrected major stage, including the observed discrepancy, the likely error source, the confirmed correction, and the resulting impact on downstream dose behavior.

#### Scenario: Stage summary is recorded after a confirmed fix
- **WHEN** a correction is applied to a major numerical stage
- **THEN** the change records what was wrong, what was changed, and how the dose behavior moved after the correction

#### Scenario: Unproven discrepancies remain explicit
- **WHEN** a stage cannot yet be proven correct from fixture evidence
- **THEN** the stage summary keeps the error source or status explicit rather than hiding it behind a later aggregate result

### Requirement: Geometry and transport stages SHALL be corrected before downstream accumulation is considered converged
The system SHALL treat CPB geometry, reference-plane geometry, and WEPL generation as upstream prerequisites for downstream convergence. Sigma transport, IDD/CIDD behavior, superposition, and final dose writeback SHALL NOT be declared converged unless the upstream geometry and transport interpretation are first validated.

#### Scenario: Bragg peak depth debugging starts upstream
- **WHEN** the Bragg peak position is incorrect
- **THEN** the investigation checks CPB, reference-plane, and WEPL/depth-axis stages before accepting downstream convolution or writeback-only fixes

#### Scenario: Writeback-only fixes do not mask upstream errors
- **WHEN** a dose writeback orientation issue is found
- **THEN** the system still validates upstream geometry and transport stages before treating the overall numerical path as corrected

### Requirement: Energy-layer and depth-dose behavior SHALL remain consistent through LUT and IDD/CIDD mapping
The system SHALL keep energy layers, LUT lookup, peak-depth interpretation, and IDD/CIDD sampling numerically consistent so that each layer contributes at the intended physical depth and the resulting depth-dose behavior does not collapse or shift spuriously.

#### Scenario: Energy-layer mapping matches intended depth behavior
- **WHEN** a fixture beam with multiple energy layers is evaluated
- **THEN** the corresponding LUT and IDD/CIDD stages preserve the intended layer ordering and physical depth relationship

#### Scenario: Peak-depth errors remain attributable
- **WHEN** a peak shift is observed after geometry has been validated
- **THEN** the investigation isolates LUT, peak-depth scaling, or IDD/CIDD sampling as explicit downstream candidates rather than treating the shift as unexplained

### Requirement: Final dose-grid output SHALL remain numerically aligned between wrapper fixture and pybind entry
The system SHALL preserve one corrected wrapper-backed numerical path and confirm that the public pybind final-dose entry reflects the same corrected dose-grid behavior, including depth orientation and writeback interpretation.

#### Scenario: Wrapper and pybind share corrected dose-grid semantics
- **WHEN** the corrected main path is validated from both the wrapper fixture and Python
- **THEN** both entrypoints reflect the same corrected depth orientation and dose-grid interpretation

#### Scenario: Python validation follows wrapper convergence
- **WHEN** a stage is corrected in the wrapper-backed path
- **THEN** the pybind-facing final-dose path is rechecked against the same fixture expectations rather than tuned independently
