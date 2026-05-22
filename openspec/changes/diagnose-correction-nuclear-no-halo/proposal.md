## Why

Stop the current halo investigation thread and open a narrower change focused on the actual failure mode now observed in plan runs: when `nuclear_correction` is enabled, the primary branch is reduced by `nucWeight`, but no visible halo/broadening is added back to the final dose. The correction-on dose can therefore be lower than correction-off, which means the nuclear branch is not contributing at the expected magnitude or is not reaching the final dose volume.

The `0..14` support/cropping issue is explicitly out of scope here. The current question is whether the correction-controlled nuclear workflow differs from `RayTraceDicom-main` in a way that prevents new nuclear dose from being generated or transferred.

Upstream has a clear correction-on flow:

```text
cumul IDD + nucWeight/nucSqSigma
        |
        v
fillIddAndSigma
  primary: res    = (1 - nucWeight) * rayWeight * dIDD / mass
  nuclear: nucRes = nucWeight * nucRayWeight * dIDD / (mass * spotDist^2)
        |
        v
primary superposition + nuclear superposition
        |
        v
primTransfDiv + nucTransfDiv add into final dose
```

The local path must prove each of those stages is active and upstream-equivalent before any final-dose broadening conclusion is trusted.

## What Changes

- Create a fresh OpenSpec change dedicated to correction-on nuclear flow parity with `RayTraceDicom-main`.
- Compare local code against upstream for:
  - nuclear LUT loading and texture coordinates
  - physical PB / halo lattice construction
  - `nucIdcs`, `nucRayWeights`, `nucMemStep`, and `spotDist` semantics
  - `NUC_IDD -> SUPERP_OUTPUT_NUC_BEV -> nucTransfDiv` handoff
- Define diagnostics that identify whether nuclear dose disappears during IDD fill, nuclear superposition, or final nuclear transfer.
- Keep `0..14` clipping/cropping out of scope except as a known confounder when interpreting final-dose support.

## Capabilities

### New Capabilities

- `correction-nuclear-halo-parity`: Diagnose and restore upstream-equivalent correction-on nuclear dose generation and transfer.

## Impact

- Affected code under investigation:
  - `src/algorithms/idd_sigma.cu`
  - `src/algorithms/complete_superposition.cu`
  - `src/core/raytracedicom_wrapper.cu`
  - `src/bindings/raytracedicom_pybind.cpp`
  - `src/utils/energy_reader.cpp`
- Upstream reference:
  - `RayTraceDicom-main/RayTraceDicom-main/src/kernel_wrapper.cu`
  - `RayTraceDicom-main/RayTraceDicom-main/src/fill_idd_and_sigma_params.*`
  - `RayTraceDicom-main/RayTraceDicom-main/src/energy_reader.cpp`
- Runtime behavior:
  - Only `nuclear_correction=true` should be affected by the eventual implementation.
