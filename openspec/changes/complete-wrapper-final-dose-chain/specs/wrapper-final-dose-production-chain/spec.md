## ADDED Requirements

### Requirement: CarbonPBS final-dose compatibility entries SHALL use the live wrapper path
The system SHALL complete the CarbonPBS-style final-dose production chain by routing `cuFinalDose`, `calcDose`, and any wrapper-backed final-dose compatibility entry through the existing `subsecondWrapper(...)` execution path. The completed chain SHALL preserve the current wrapper-centric control flow and SHALL NOT introduce a parallel final-dose implementation.

#### Scenario: Final-dose entry uses one wrapper path
- **WHEN** a caller invokes the public CarbonPBS-style final-dose API
- **THEN** the implementation routes the normalized inputs into the existing wrapper path rather than a separate computation path

#### Scenario: Wrapper structure remains primary
- **WHEN** the final-dose production chain is completed
- **THEN** the wrapper keeps its current basic form and the change only fills missing main-chain preparation or integration steps

### Requirement: The production entry SHALL build a wrapper-compatible final-dose contract
The system SHALL ensure that the public final-dose entry prepares the same effective runtime contract that the existing wrapper integration fixture relies on, including beam metadata, ROI normalization, CT or body placeholder preparation, LUT or WEQ alignment, and any table-path handling required for the live wrapper to execute correctly.

#### Scenario: Production entry completes missing handoff state
- **WHEN** CarbonPBS-style inputs are passed into the final-dose pybind entry
- **THEN** the binding layer produces wrapper-compatible beam, grid, LUT, WEQ, and runtime dependency state without requiring a second implementation

#### Scenario: Table path is callable from public entry
- **WHEN** the caller provides a valid tables directory using either a relative or absolute path form
- **THEN** the final-dose production entry resolves the required RTD tables without relying on undocumented path-string formatting

### Requirement: The pybind final-dose API SHALL be callable and produce final-dose output semantics
The system SHALL provide a callable Python-facing final-dose API that accepts CarbonPBS-style arrays, preserves the public compatibility entry names, and writes final-dose results through the wrapper-backed chain into the expected output buffer or returned array semantics.

#### Scenario: cuFinalDose writes final dose
- **WHEN** Python code invokes `cuFinalDose` with a valid output grid and valid CarbonPBS-style input tensors
- **THEN** the call executes the wrapper-backed final-dose chain and writes the dose result into the provided final-dose output grid

#### Scenario: calcDose remains wrapper-backed
- **WHEN** Python code invokes `calcDose`
- **THEN** the call produces final-dose output through the same completed wrapper-backed chain used by the other public final-dose compatibility entries

### Requirement: The completed chain SHALL be verified against the repository fixture and the public entry
The system SHALL verify the finished production chain using both the existing wrapper integration fixture and a pybind-facing final-dose invocation so that the repository proves both the internal wrapper path and the public callable entry.

#### Scenario: C++ fixture remains the reference path
- **WHEN** the completed chain is validated
- **THEN** `src/tests/wrapper_integration_test.cu` is used as the reference executable fixture for the direct wrapper path

#### Scenario: Python-facing final-dose entry is validated
- **WHEN** the completed chain is validated
- **THEN** the repository includes a verification path that invokes the public pybind final-dose API with realistic fixture inputs and confirms that the entry is callable
