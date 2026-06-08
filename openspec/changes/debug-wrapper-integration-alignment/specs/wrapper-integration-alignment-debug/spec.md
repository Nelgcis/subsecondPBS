## ADDED Requirements

### Requirement: The system SHALL use `wrapper_integration_test.cu` as the baseline integration oracle
The system SHALL treat `src/tests/wrapper_integration_test.cu` as the baseline builder for this debugging workflow. The diagnostic comparison SHALL start from the test-loaded inputs, including any geometry or metadata overrides that the test applies before calling the live wrapper.

#### Scenario: Baseline capture begins from the integration test
- **WHEN** the diagnostic workflow runs on the wrapper integration fixture
- **THEN** the baseline side of every reported comparison comes from the values built by `src/tests/wrapper_integration_test.cu` before `subsecondWrapper(...)` executes

### Requirement: The system SHALL instrument the existing wrapper path without creating a parallel implementation
The system SHALL add staged debug checkpoints inside the current `src/core/raytracedicom_wrapper.cu` execution path. The debugging workflow SHALL preserve the wrapper's basic structure, SHALL not introduce a second wrapper implementation, and SHALL not refactor the main flow as a prerequisite for diagnosis.

#### Scenario: Single-path wrapper debugging
- **WHEN** diagnostics are enabled for the wrapper integration path
- **THEN** the system records checkpoint data from the live wrapper execution path rather than from a mirrored or replacement implementation

### Requirement: The system SHALL expose geometry, grid, and ROI normalization checkpoints
The system SHALL capture and compare the raw and normalized state for beam geometry and grid-related inputs, including `beamDirection`, `bmxdir`, `bmydir`, `source`, `sad`, `sourceDist`, `refPlaneZ`, `ctGrid`, `doseGrid`, and ROI encoding. If the current code cannot prove that a normalization is correct, the comparison result SHALL be marked `unknown`.

#### Scenario: Geometry and grid mismatch review
- **WHEN** the wrapper resolves beam basis vectors, source-distance semantics, grid metadata, or ROI indices
- **THEN** the diagnostic output states the input value, the normalized wrapper value, the transform rule that was applied, and whether the result is aligned, transformed-but-proven, mismatched, or `unknown`

### Requirement: The system SHALL expose spot, subspot, LUT, and energy-layer mapping checkpoints
The system SHALL capture and compare the mapping stages for `idbeamxy`, raw spot lattice reconstruction, WEQ or BEV transfer, `layer_energy`, `energy_list`, longitudinal cutoffs, `profile_data`, `beam_para_data`, `subspotData`, and exact convolution inputs. The diagnostic output SHALL preserve raw-versus-normalized state wherever decode ambiguity or row-selection ambiguity exists.

#### Scenario: Spot and LUT mapping review
- **WHEN** the wrapper derives lattice indices, row selectors, cutoff depths, or exact convolution inputs from spot or LUT-backed data
- **THEN** the diagnostic output identifies the stage, the raw source values, the normalized values used by the wrapper, and any confirmed mismatch or unresolved ambiguity

### Requirement: The system SHALL localize and rank likely causes of abnormal dose and Bragg peak displacement
The system SHALL produce a stage-ordered mismatch-localization report that identifies the first confirmed divergence points along the live wrapper path and ranks the likely causes of abnormal dose or Bragg peak displacement using observed evidence. If causality cannot be proven from the current run, the unproven portion SHALL be marked `unknown`.

#### Scenario: Diagnostic conclusion review
- **WHEN** a diagnostic wrapper run completes
- **THEN** the report lists the confirmed mismatch stages in execution order and provides an evidence-based ranking of the mismatch points most likely to explain the dose anomaly or Bragg peak offset

### Requirement: The debugging workflow SHALL preserve current algorithm behavior
The system SHALL keep the diagnostic workflow read-only with respect to the numerical wrapper algorithm. Enabling checkpoint capture SHALL not require a different kernel path, a different data-flow path, or a wrapper rewrite.

#### Scenario: Behavior-preserving diagnostics
- **WHEN** the debugging workflow is enabled
- **THEN** the wrapper still executes the same main path and the change scope remains observational rather than algorithm-changing
