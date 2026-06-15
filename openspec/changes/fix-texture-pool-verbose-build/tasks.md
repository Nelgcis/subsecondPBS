## 1. Fix Compile Dependency

- [x] 1.1 Add the existing shared debug-tools include to `src/utils/texture_pool.cu` using the local include style used by neighboring utility `.cu` files.
- [x] 1.2 Confirm the file still uses `rtdVerboseFineTiming()` for texture pool init/cleanup timing and does not introduce duplicate verbose parsing or a new environment flag.

## 2. Verify Build Path

- [x] 2.1 Run the legacy `make` command path that previously failed and confirm compilation proceeds past `src/utils/texture_pool.cu` without undefined `rtdVerboseFineTiming` errors.
- [x] 2.2 If later unrelated build errors appear after `texture_pool.cu` compiles, record them separately without expanding this change beyond the texture-pool verbose helper fix.
  Later build state: `make` now reaches `src/bindings/raytracedicom_pybind.cpp` and fails on missing `Python.h`; that is a separate pybind/Python-dev environment issue, not a `texture_pool.cu` verbose helper error.
