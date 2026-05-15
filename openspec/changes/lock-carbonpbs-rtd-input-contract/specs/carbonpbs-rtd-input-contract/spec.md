## ADDED Requirements

### Requirement: Contract locking SHALL not change algorithm behavior
The system SHALL lock and document the CarbonPBS to test-data to RayTraceDicom input contract without changing the numerical dose algorithm, kernel-selection behavior, or physical interpretation of an already valid execution path. Contract work in this change SHALL be limited to naming, shape definition, axis definition, metadata definition, normalization rules, validation rules, and explicit `unknown` classification.

#### Scenario: Contract-lock review
- **WHEN** a reviewer inspects the change scope and resulting implementation plan
- **THEN** the change only defines and enforces input-contract semantics and does not introduce algorithmic dose changes

### Requirement: The system SHALL publish a variable-by-variable contract matrix
The system SHALL publish a source-anchored contract matrix for the CarbonPBS integration chain. For each exported or consumed field, the matrix SHALL identify the producer name, observed shape, current legacy CarbonPBS consumer, current RayTraceDicom consumer, canonical future contract, and status classification. If the current repository cannot prove a field's meaning, axis, or unit, the matrix SHALL label that field `unknown`.

#### Scenario: Variable-contract review
- **WHEN** a reviewer checks any input field used by `dosecal.py`, `test_data`, `carbonPBS`, or the RTD wrapper
- **THEN** the contract matrix states the field meaning, shape, consumer mapping, and whether the current repository proves or does not prove consistency

### Requirement: The system SHALL define the canonical `subspotData` contract
The system SHALL define `subspotData` channel semantics, tensor dimensions, flattening rules, indexing rules, and mapping rules into the RayTraceDicom integration path. The contract SHALL state how the legacy CarbonPBS texture layout maps into the current RTD exact-convolution path and SHALL define what metadata is required when the `subspotData` row axis is not identical to `layer_energy`.

#### Scenario: `subspotData` contract review
- **WHEN** a reviewer checks `subspotData`
- **THEN** the documentation states the meaning of all five channels, the row or subspot or channel ordering, the texture indexing rules, and the required normalization before the RTD wrapper consumes it

### Requirement: The system SHALL normalize ROI inputs to one canonical internal representation
The system SHALL define one canonical internal ROI representation for the integration path. The contract SHALL state which Python-side forms are accepted at the boundary, how they are normalized, how linearization is computed, and which representation future `calDoseSubsecond` code SHALL use internally.

#### Scenario: ROI input review
- **WHEN** a caller provides ROI indices as flat xyz triplets or as shape `(N,3)` or `(3,N)`
- **THEN** the contract states exactly how those forms are converted to the canonical internal ROI representation and which shape is used internally

### Requirement: The system SHALL define beam geometry and grid metadata contracts
The system SHALL define the geometry contract for `beamDirection`, `bmxdir`, `bmydir`, `source`, `sad`, and reference-plane semantics, and SHALL define the grid metadata contract for `doseGrid` and `ctGrid` `dims`, `corner`, and `resolution`. The contract SHALL state coordinate frames, axis order, units when proven, the mapping between CarbonPBS exports and RTD wrapper fields, and the corresponding Python numpy shape order requirements. Any unresolved relationship SHALL be marked `unknown`.

#### Scenario: Geometry-and-grid review
- **WHEN** a reviewer checks any geometry or grid field used by the compatibility chain
- **THEN** the contract states the field definition, coordinate frame, axis order, and mapping into the current RTD wrapper or marks the unresolved part `unknown`

### Requirement: The system SHALL verify fixture parity and classify integration risk
The system SHALL compare `dosecal.py` or `test_data` outputs with the actual legacy CarbonPBS inputs and the actual RayTraceDicom wrapper reads. The result SHALL classify each field as consistent, ambiguous, rename-needed, shape-risk, unit-risk, order-risk, or `unknown`, and SHALL publish a checklist of the high-risk misalignment points that future `calDoseSubsecond` work must resolve or explicitly preserve.

#### Scenario: Fixture-parity review
- **WHEN** a reviewer checks the fixture-parity matrix and readiness checklist
- **THEN** the reviewer can see which fields already match, which fields drift, which fields require explicit shape or naming work, and which fields cannot yet be proven consistent

### Requirement: The system SHALL define the future `calDoseSubsecond` input contract
The system SHALL publish a canonical future input contract for `calDoseSubsecond`. That contract SHALL identify the required normalized fields, the required metadata for row axes and buffer order, the canonical internal ROI representation, the required grid and geometry definitions, and the rule that ambiguous or unproven inputs SHALL not be silently inferred.

#### Scenario: Future-implementation review
- **WHEN** a future `calDoseSubsecond` implementation is planned
- **THEN** the implementation can follow the published canonical contract and can reject or flag any input state that the contract marks ambiguous or `unknown`
