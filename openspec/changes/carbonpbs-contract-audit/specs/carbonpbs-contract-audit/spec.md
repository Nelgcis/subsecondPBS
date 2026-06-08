## ADDED Requirements

### Requirement: The change SHALL remain audit-only
The `carbonpbs-contract-audit` change SHALL deliver source-anchored analysis artifacts only and SHALL not modify runtime code, bindings, CUDA kernels, tests, or table contents in the repository.

#### Scenario: Review of change contents
- **WHEN** a reviewer inspects the change contents and repository diff
- **THEN** only OpenSpec artifacts are modified and the audited source files remain unchanged

### Requirement: The audit dossier SHALL cover the requested repository scope
The audit dossier SHALL analyze `carbonPBS/cudaCalDose.cpp`, `carbonPBS/deviceCalDose.cu`, `src/core/raytracedicom_wrapper.cu`, `src/algorithms/convolution.cu`, `src/bindings/raytracedicom_pybind.cpp`, `test_data`, and `/tables`, and SHALL compare the current compatibility path against `RayTracedicom_main` reference behavior.

#### Scenario: Scope review
- **WHEN** the dossier is reviewed section by section
- **THEN** every requested repository area is covered and the upstream RTD comparison basis is named explicitly

### Requirement: The audit dossier SHALL define variable, coordinate, and texture contracts
The audit dossier SHALL contain a variable contract table, a coordinate-system table, and a `subspotData` texture layout table. These sections SHALL state actual shapes, axes, units, flattening rules, and current consumer paths for the audited runtime surfaces.

#### Scenario: Contract table review
- **WHEN** a reviewer inspects the contract sections
- **THEN** the reviewer can trace each major CarbonPBS input from exported form to its current in-repo consumer without consulting implementation plans

### Requirement: The audit dossier SHALL map exported test data to actual consumers
The audit dossier SHALL map the `dosecal.py` and `test_data` fields used by the CarbonPBS fixture to their actual consumers in the legacy CarbonPBS path, the current pybind compatibility path, the wrapper, the convolution path, and `/tables`, and SHALL identify fields that are ignored, collapsed, remapped, or used only by test harnesses.

#### Scenario: Consumer mapping review
- **WHEN** a reviewer checks any exported fixture field
- **THEN** the dossier identifies whether that field is consumed, transformed, ignored, or used only for testing and names the current consumer path

### Requirement: The audit dossier SHALL enumerate upstream deviations
The audit dossier SHALL include a complete deviation list for the audited scope versus `RayTracedicom_main`, including API, data-model, geometry, tracing, convolution, and table-loading differences that materially change the CarbonPBS compatibility contract.

#### Scenario: Upstream drift review
- **WHEN** a reviewer compares the current repository against the named upstream RTD reference files
- **THEN** the dossier provides a source-anchored deviation entry for each material behavioral difference in scope

### Requirement: The audit dossier SHALL rank likely Bragg peak error causes
The audit dossier SHALL include a ranked list of likely causes of Bragg peak position error, and each ranked item SHALL tie the hypothesis to the current source contract rather than to an implementation proposal.

#### Scenario: Bragg peak diagnosis review
- **WHEN** a reviewer inspects the ranked cause list
- **THEN** the reviewer can see which current contract mismatches are most likely to move the peak position and why they are ranked in that order
