## Why

Some plans trigger many primary superposition radius overflow warnings in the
subSecond RTD-backed path:

```text
Warning: Primary superposition required radius larger than max radius; clamping to max radius
```

Other plans do not. The immediate performance impact is plan-dependent because
overflow tiles are clamped into the `MAX_SUPERP_RADIUS=32` bucket, which makes
the most expensive superposition kernel variant participate in the calculation.

The project needs a focused diagnostic change before optimization work:

- identify which plan, CT/WEQ, geometry, energy-layer, and beam-parameter factors
  drive small `rSigmaEff` and radius overflow
- compare the CarbonPBS direct voxel/spot model with subSecond's RTD
  ray-tracing + IDD/sigma + superposition model
- avoid creating new plans just to reproduce the issue; derive explanations from
  existing plan inputs and per-layer diagnostics
- separate true physics/geometry causes from implementation artifacts such as
  unit conversion, ray spacing, WEQ origin, sliced LUT reconstruction, and
  radius-bucket scheduling

## What Changes

This change first establishes a read-only investigation and task plan. The
implementation work, when started later, should add diagnostics and guardrails
around the existing wrapper path rather than replacing the algorithm.

Initial scope:

- Map the radius-overflow chain from `FillIddAndSigmaParams` and
  `fillIddAndSigmaKernel` through `tileRadCalc`.
- Contrast CarbonPBS' direct `weqDepth/profileData/beamParaData` dose evaluation
  with subSecond's RTD `sigmaSq -> rSigmaEff -> tile radius` path.
- Define per-plan and per-energy-layer metrics that can explain why only some
  plans overflow.
- Prioritize radius-overflow diagnosis before broader resource scheduling and
  CUDA 12.1 optimization.

Deferred but tracked scope:

- Explain why subSecond is currently slower than CarbonPBS beyond the radius
  overflow path, and identify which runtime costs are cleanup debt versus
  algorithmic trade-offs.
- Audit whether there are latent numerical error-accumulation paths in
  subSecond.
- Explore CUDA 12.1-era performance options after correctness and diagnostics
  are trustworthy.

## Non-Goals

- Do not create a new plan just to force overflow.
- Do not hide overflow by blindly raising `MAX_SUPERP_RADIUS` or post-processing
  dose.
- Do not introduce a second dose engine.
- Do not optimize kernels before the root factors for overflow are measurable.

## Impact

Likely affected investigation surfaces:

- `src/algorithms/idd_sigma.cu`
- `include/algorithms/fill_idd_and_sigma_params.cuh`
- `src/algorithms/complete_superposition.cu`
- `src/core/raytracedicom_wrapper.cu`
- `src/bindings/raytracedicom_pybind.cpp`
- CarbonPBS comparison source:
  `/home/gadolinite/CASHIM_HL/subsecond/raytracedicom_pybind_stage/patch10_mod_20260410/carbonPBS`

