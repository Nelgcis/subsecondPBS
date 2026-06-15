## ADDED Requirements

### Requirement: Utility translation units include shared verbose helpers before use
The build SHALL compile utility CUDA translation units that call RTD verbose helper functions by including the shared debug-tools interface that defines those helpers.

#### Scenario: Texture pool verbose helper resolves during compilation
- **WHEN** `src/utils/texture_pool.cu` is compiled by the legacy `make` build
- **THEN** references to `rtdVerboseFineTiming()` MUST resolve without undefined identifier errors

#### Scenario: Texture pool timing remains controlled by RTD verbose mode
- **WHEN** texture pool init or cleanup timing messages are compiled
- **THEN** they MUST continue to use the existing `rtdVerboseFineTiming()` behavior rather than a duplicate or texture-pool-specific verbose check

### Requirement: Legacy make build verifies the texture pool fix
The change SHALL be verified through the same `make` build path that exposed the failure, at least far enough to prove `src/utils/texture_pool.cu` no longer fails on unresolved verbose helpers.

#### Scenario: Build proceeds past texture pool compilation
- **WHEN** `make` is run after the fix
- **THEN** the build MUST proceed past `src/utils/texture_pool.cu` without the `identifier "rtdVerboseFineTiming" is undefined` error
