## Why

The legacy `make` build currently fails in `src/utils/texture_pool.cu` because it calls `rtdVerboseFineTiming()` without including the header that defines the inline verbose helpers. This blocks rebuilding the patched tree even though the failure is only a local dependency/include issue.

## What Changes

- Fix the `texture_pool.cu` build break by wiring it to the existing RTD debug/verbose helper interface.
- Keep verbose behavior unchanged: texture-pool timing messages remain controlled by `RTD_VERBOSE` / `rtdVerboseFineTiming()`.
- Verify the legacy `make` path reaches past `src/utils/texture_pool.cu` without the undefined identifier error.
- Do not introduce a new verbose flag, duplicate helper implementation, or change runtime texture-pool behavior.

## Capabilities

### New Capabilities
- `build-verification`: The repository build surfaces should compile utility translation units that use shared RTD verbose helpers without unresolved identifiers.

### Modified Capabilities

## Impact

- Affected code: `src/utils/texture_pool.cu`, and possibly only include ordering around existing debug utilities.
- Affected build path: the repository `Makefile` / legacy `make` compilation path shown by the failing command.
- APIs: none.
- Runtime behavior: no intended change except preserving existing verbose-controlled texture pool timing logs.
