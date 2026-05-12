## ADDED Requirements

### Requirement: CarbonPBS-style final-dose entries SHALL use the live RTD wrapper path
The system SHALL complete the production final-dose chain by routing CarbonPBS-style pybind entries, including `cuFinalDose`, `calcDose`, and any `cuCalDose`-style compatibility entry that still exposes final-dose behavior, through the existing `raytracedicom_wrapper` execution path. The completed chain SHALL preserve the current basic form of `src/core/raytracedicom_wrapper.cu` and SHALL NOT introduce a parallel final-dose implementation.

#### Scenario: Public final-dose entry uses one RTD path
- **WHEN** a caller invokes a supported CarbonPBS-style final-dose pybind entry with valid inputs
- **THEN** the implementation marshals those inputs into wrapper-compatible RTD structures and executes the existing RTD main flow rather than a second computation path

#### Scenario: Wrapper structure remains primary
- **WHEN** the production final-dose chain is completed
- **THEN** `src/core/raytracedicom_wrapper.cu` remains the primary numerical producer and the change only fills missing handoff or runtime-link steps needed to reach it

### Requirement: The production chain SHALL lock the CarbonPBS-to-RTD input contract
The system SHALL define and enforce the effective runtime contract between repository `test_data`, CarbonPBS-style pybind arguments, and wrapper-consumed RTD structures. This contract SHALL cover geometry variables, dose/CT grid fields, ROI indexing, layer energies, layer spot counts, spot positions, spot weights, subspot data, LUT settings, and any related storage-order or unit conventions needed to execute the wrapper correctly.

#### Scenario: Pybind entry normalizes wrapper-facing inputs
- **WHEN** CarbonPBS-style arrays and scalars are passed into the public pybind final-dose API
- **THEN** the binding layer normalizes their shape, ordering, and compatibility defaults into the wrapper-facing contract proven by `tests/wrapper_integration_test`

#### Scenario: Runtime contract stays aligned with fixture inputs
- **WHEN** the same beam case is prepared through repository `test_data` and through the direct wrapper fixture path
- **THEN** the production entry consumes the corresponding geometry, grid, ROI, energy-layer, subspot, and LUT variables under the same effective contract required by the wrapper path

### Requirement: WEQ-backed transport SHALL drive the completed RTD main path
The system SHALL preserve tracing behavior aligned with `raytracedicom-main`, while ensuring that transport-driving calculations in the integrated production path consume WEQ-backed data rather than relying on CT-derived transport semantics. If wrapper compatibility still requires CT-shaped buffers at entry, those buffers SHALL serve only as boundary-compatible placeholders and SHALL NOT redefine the intended WEQ-backed transport contract.

#### Scenario: WEQ data remains authoritative for transport handoff
- **WHEN** the production final-dose path prepares inputs for RTD tracing and dose computation
- **THEN** WEQ data provides the transport-driving information required by the completed integration path, with any CT placeholder restricted to compatibility support

#### Scenario: Tracing semantics remain aligned with raytracedicom-main
- **WHEN** the final-dose chain executes through the wrapper
- **THEN** the tracing stage follows the repository's RTD main-path semantics rather than a simplified or alternate transport implementation

### Requirement: Public pybind final-dose APIs SHALL remain callable and write final dose with consistent semantics
The system SHALL keep the supported pybind compatibility entry names callable, SHALL resolve RTD reference tables from both relative and explicit `tables` directory inputs, and SHALL write final dose through the wrapper-backed path into the expected output grid semantics.

#### Scenario: Table path resolves for relative and explicit inputs
- **WHEN** a caller provides a valid `tables` directory as either a relative path or an explicit path
- **THEN** the public final-dose path resolves the required RTD reference tables without depending on undocumented trailing-separator formatting

#### Scenario: Final dose is written through the public API
- **WHEN** Python code invokes a supported CarbonPBS-style final-dose entry with a valid output dose grid and valid runtime inputs
- **THEN** the call executes the wrapper-backed path and writes the produced dose values into the expected output buffer or returned dose grid

#### Scenario: Public entry does not silently succeed with placeholder-zero output
- **WHEN** Python code invokes a supported CarbonPBS-style final-dose entry with positive spot weights
- **THEN** the pybind path either returns a non-empty dose grid produced by the wrapper-backed chain or raises an error instead of silently succeeding with an all-zero placeholder result

### Requirement: The completed chain SHALL be verified through both fixture and pybind entrypoints
The system SHALL verify the completed production chain using the existing wrapper integration fixture as the direct RTD reference path and a pybind-facing final-dose invocation that exercises realistic repository `test_data`.

#### Scenario: Wrapper fixture remains the direct reference path
- **WHEN** the completed chain is validated
- **THEN** `tests/wrapper_integration_test` is used as the direct reference executable for the wrapper-facing path

#### Scenario: Wrapper fixture emits a pybind-equivalent dose-grid artifact
- **WHEN** `tests/wrapper_integration_test` writes dose output for a repository beam fixture
- **THEN** it produces a bin artifact whose stored 3D dose-grid layout matches the public pybind external dose-grid semantics for direct validation

#### Scenario: Pybind entry is validated with repository fixture data
- **WHEN** the completed chain is validated
- **THEN** the repository includes a verification path that invokes the public pybind final-dose API with realistic `test_data` inputs and confirms that the entry is callable or documents any remaining environment-only blocker explicitly

#### Scenario: Repository provides a direct Python fixture runner
- **WHEN** the completed chain is validated from Python
- **THEN** the repository includes a Python-side runner that loads fixture inputs, invokes the public pybind final-dose entry, and writes a directly verifiable 3D dose-grid bin artifact
