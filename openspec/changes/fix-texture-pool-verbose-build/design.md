## Context

The legacy `make` build compiles `src/utils/texture_pool.cu` directly with include paths for `include/`, `include/algorithms`, `include/core`, and `include/utils`. That translation unit calls `rtdVerboseFineTiming()` in `initializeTexturePool()` and `cleanupTexturePool()`, but it does not include `include/utils/debug_tools.h`, where the inline verbose helpers are defined.

Other utility and algorithm translation units that use the same helper already include `debug_tools.h`. The texture pool should follow that local pattern rather than defining a duplicate helper or adding a new logging switch.

## Goals / Non-Goals

**Goals:**
- Restore the legacy `make` build past `src/utils/texture_pool.cu`.
- Preserve the existing RTD verbose semantics driven by `RTD_VERBOSE` and `rtdVerboseFineTiming()`.
- Keep the fix scoped to the missing dependency/include relationship.

**Non-Goals:**
- Change texture pool allocation, caching, cleanup, or statistics behavior.
- Add a new texture-pool-specific debug flag.
- Refactor the broader debug-tools interface or Makefile structure.
- Address unrelated warnings or later compilation failures that appear after this file compiles.

## Decisions

Use the existing shared debug-tools header in `texture_pool.cu`.

Rationale: `rtdVerboseFineTiming()` is an inline helper in `include/utils/debug_tools.h`, and multiple neighboring `.cu` files already include that header before calling it. Adding the include keeps behavior consistent and avoids duplicating environment parsing.

Alternative considered: replace the calls with direct `std::getenv("RTD_VERBOSE")` checks. Rejected because it would fork verbose semantics and make future changes harder to keep consistent.

Alternative considered: remove the timing prints from texture-pool init/cleanup. Rejected because it changes existing debug behavior unnecessarily.

## Risks / Trade-offs

- [Header include path mismatch under `make`] -> Use the same relative include style already used by other `src/utils/*.cu` files.
- [Fix reveals later build errors] -> Treat any later errors as separate issues unless they are directly caused by the new include.
- [Accidental runtime behavior change] -> Limit implementation to dependency wiring and compile verification only.
