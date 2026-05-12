## ADDED Requirements

### Requirement: Direct wrapper final-dose execution SHALL consume the exported test-data contract

The system SHALL allow `src/tests/wrapper_integration_test.cu` to consume the exported `test_data/` variables corresponding to `doseCal.py -> cuCalDose/cuFinalDose` inputs and compute a `dosegrid` through the live RTD wrapper-backed path without requiring pybind.

#### Scenario: Direct wrapper test uses exported CarbonPBS-style inputs

- **WHEN** the direct wrapper test is pointed at a valid `test_data` case
- **THEN** the test loads the exported beam, grid, WEQ, LUT, and runtime arrays and computes final dose through the live wrapper-backed RTD chain

#### Scenario: Direct wrapper output remains a dosegrid result

- **WHEN** the direct wrapper test executes successfully
- **THEN** the result is a direct wrapper-produced `dosegrid`, not a pybind-only compatibility artifact

### Requirement: Final-dose pybind compatibility entries SHALL remain wrapper-backed

The system SHALL keep `cuCalDose`, `cuFinalDose`, and equivalent final-dose compatibility entries on the same RTD wrapper-backed final-dose path used by the direct wrapper test.

#### Scenario: Final-dose public entry and direct wrapper share semantics

- **WHEN** a caller invokes a supported RTD final-dose compatibility entry
- **THEN** the binding layer prepares the same effective contract meanings used by the direct wrapper path and routes the call into the wrapper-backed RTD final-dose chain

### Requirement: The `.so` surface SHALL expose honest operator-family semantics

The system SHALL expose the required CarbonPBS-style operator-family entry names through the compiled shared object, and each entry SHALL either provide a real supported implementation or an explicit unsupported/capability-gated behavior. The system SHALL NOT silently substitute zero-valued CSC outputs for a requested CSC-style operator.

#### Scenario: Unsupported CSC-style behavior is explicit

- **WHEN** a caller invokes a CSC-style or norm-style entry that is not yet truly implemented in RTD
- **THEN** the call fails explicitly or reports unsupported capability, rather than returning zero-filled outputs as if they were valid

#### Scenario: Supported operator-family entries are callable by name

- **WHEN** downstream code imports the RTD shared object
- **THEN** the expected CarbonPBS-style operator-family symbol names are present according to the staged support matrix for the change

### Requirement: Future RTD operator-family support SHALL preserve spot identity

The system SHALL implement true RTD-native CSC and row-norm behavior only through a spot-preserving operator path rather than by deriving those outputs from already-aggregated final-dose intermediates.

#### Scenario: CSC-style support is based on spot-preserving transport

- **WHEN** RTD-native CSC or row-norm support is added
- **THEN** the implementation preserves spot identity through the relevant transport stage instead of inferring CSC semantics from aggregated final-dose data
