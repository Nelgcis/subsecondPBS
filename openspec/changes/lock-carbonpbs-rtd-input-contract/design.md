## Context

This change defines how to lock the CarbonPBS to test-data to RayTraceDicom input contract before implementing `calDoseSubsecond`. The current repository already contains one runtime chain, but not one stable contract:

- `tps_py/dosecal.py` and `test_data` export one set of array names and shapes
- `carbonPBS/cudaCalDose.cpp` and `carbonPBS/deviceCalDose.cu` consume a legacy CarbonPBS contract
- `src/bindings/raytracedicom_pybind.cpp` adapts those arrays into a custom `RTDBeamSettings` and `RTDEnergyStruct`
- `src/core/raytracedicom_wrapper.cu` consumes that adapted contract using additional geometry, unit, and lattice heuristics
- `src/algorithms/convolution.cu` preserves the upstream RTD convolution kernels but adds CarbonPBS-specific subspot and CPB bridging logic

This creates immediate implementation risk for `calDoseSubsecond` because several fields are already known to drift:

- `subspot_data` is exported on a LUT energy axis but may be consumed on actual layer energies after remap
- ROI inputs can arrive as xyz triplets or 2D arrays but are normalized to linear indices only after pybind entry
- grid metadata is expressed in `(x,y,z)` vectors, while some Python arrays and wrapper return values use `(Z,Y,X)` ndarray order
- `idbeamxy` is exported as WEQ-lattice index-like coordinates but is decoded in multiple ways inside the wrapper
- the current compatibility path synthesizes some RTD inputs from `/tables` even though those values are not explicit CarbonPBS exports

Evidence base for the locked contract:

- `carbonPBS/cudaCalDose.cpp`
- `carbonPBS/deviceCalDose.cu`
- `src/bindings/raytracedicom_pybind.cpp`
- `src/core/raytracedicom_wrapper.cu`
- `src/algorithms/convolution.cu`
- `include/core/raytracedicom_integration.h`
- `src/tests/carbonpbs_data_layout_example.cu`
- `src/tests/wrapper_integration_test.cu`
- `tps_py/dosecal.py`
- `test_data/rtd_test_case_bg800_beam925.py`
- `test_data/dose_inputs_csv/variable_name_meta.csv`
- `test_data/dose_inputs_csv/calc_required_meta.csv`
- `tables/` and `tables/generated/`
- upstream RayTraceDicom reference files in `RayTraceDicom-main (1)/RayTraceDicom-main/src/`

Observed `bg800_beam925` fixture facts that matter for contract locking:

- `layer_info.shape = (29,)`
- `layer_energy.shape = (29,)`
- `all_energies.shape = (6967,)`
- `number_particle.shape = (6967,)`
- `longitudal_cutoff.shape = (6967,)`
- `energy_list.shape = (123,)`
- `idd_data.shape = (123, 4000)`
- `profile_data.shape = (123, 800, 11)`
- `beam_para_data.shape = (123, 3)`
- `subspot_data.shape = (123, 1, 5)`
- `idbeamxy.shape = (6967, 2)`
- `calc_required_meta.csv` declares `idbeamxy is rayweq texture index-like coordinate, not direct gantry-plane physical coordinate`

## Goals / Non-Goals

**Goals:**

- Freeze one explicit, source-anchored contract for the CarbonPBS integration chain without changing algorithm behavior.
- Separate the observed current contract from the canonical future contract that `calDoseSubsecond` must obey.
- Lock the meaning, shape, axis order, coordinate frame, and normalization rule for all major inputs.
- Classify every variable as consistent, ambiguous, rename-needed, shape-risk, unit-risk, order-risk, or `unknown`.
- Produce a checklist that future implementation can use as an input gate.

**Non-Goals:**

- Changing dose math, RTD kernel selection, or physical beam modeling.
- Replacing existing data files or regenerating `/tables`.
- Solving Bragg-peak position numerically in this change.
- Guessing units or semantics that the current repository does not prove.

## Decisions

### Decision: Separate the observed contract from the locked canonical contract

The design keeps two views side by side:

- `Observed current contract`: what the checked-in code actually does today
- `Locked canonical contract`: what future `calDoseSubsecond` code must normalize to before execution

Rationale:

- The current repository already contains incompatible interpretations of some variables.
- Locking the future contract directly on top of the current code would otherwise hide the existing drift.
- This lets the change document `unknown` states without turning them into invented facts.

Alternatives considered:

- Document only the intended future contract. Rejected because it would lose the actual current mismatches that must be normalized or rejected.

### Decision: Canonicalize ROI internally as linear dose-grid indices

The canonical internal ROI representation is `roi_linear_indices`, using C-order linearization over `(Nx, Ny, Nz)`:

`linear = ix * (Ny * Nz) + iy * Nz + iz`

Boundary adapters may accept flat xyz triplets or 2D `(N,3)` or `(3,N)` arrays, but future `calDoseSubsecond` code must not keep multiple internal ROI encodings alive at once.

Rationale:

- This matches the current pybind normalization path and the wrapper's internal `roiLinearIndices` usage.
- It cleanly separates boundary compatibility from internal execution.

Alternatives considered:

- Canonicalize ROI internally as `vec3i` triplets. Rejected because the current wrapper and tests already operate on linear indices after normalization.

### Decision: Canonicalize grid metadata in `(x, y, z)` vectors and require explicit ndarray order

The locked contract keeps `dims`, `corner`, and `resolution` as `(x, y, z)` metadata vectors. Array buffers must additionally state their numpy order explicitly. The future contract forbids implicit inference from metadata alone.

Rationale:

- The current repository already mixes `(Nx,Ny,Nz)` metadata with both `(Nx,Ny,Nz)` and `(Nz,Ny,Nx)` Python arrays.
- The metadata vectors themselves are consistently treated as `(x,y,z)` in the compatibility layer.

Alternatives considered:

- Force one ndarray order everywhere immediately. Rejected in this change because the current repository does not yet implement that normalization consistently.

### Decision: Make row-axis metadata explicit for LUT-backed tensors

The locked contract treats `subspot_data`, `profile_data`, and `beam_para_data` as tensors whose first dimension must declare which axis it follows:

- `layer_energy`
- `energy_list`
- or `unknown`

If a tensor is not already aligned to actual plan layers, future normalization must make that explicit instead of silently inferring by shape.

Rationale:

- The current fixture uses 123 LUT rows for these tensors but 29 actual plan layers.
- The current pybind path remaps `subspot_data` but not every tensor the same way.

Alternatives considered:

- Preserve silent remap by row-count heuristics. Rejected because that is exactly the ambiguity the change is supposed to remove.

### Decision: Treat `bmxdir` and `bmydir` as primary basis vectors and require explicit reference-plane semantics

The locked contract treats `bmxdir` and `bmydir` as the primary beam-plane basis exported from CarbonPBS-compatible inputs. `beamDirection` is used to disambiguate sign and to validate consistency. The reference plane must be explicit in the future contract; if the current repository does not expose it directly, the status remains `unknown`.

Rationale:

- The current wrapper derives `bmZ = cross(bmxdir, bmydir)` and then fixes its sign using `beamDirection`.
- Test data does not expose an independent explicit reference-plane variable even though the wrapper assumes one.

Alternatives considered:

- Treat `beamDirection` as the primary geometry and `bmxdir` or `bmydir` as optional. Rejected because that does not match the current wrapper's actual basis construction.

### Decision: Unproven equivalence stays `unknown`

If the current repository cannot prove a unit, axis order, or semantic equivalence, the design marks it `unknown` and requires future implementation either to carry explicit metadata or to reject the input at normalization time.

Rationale:

- The user explicitly requested no invented values.
- Silent inference is one of the current failure modes.

Alternatives considered:

- Fill gaps using likely physical interpretation. Rejected because that would hide the real contract gaps.

## Variable Crosswalk And Locked Contract

Status labels used below:

- `consistent`: current producer and current consumer agree well enough to lock directly
- `ambiguous`: current code uses the field but more than one interpretation is alive
- `shape-risk`: current first-dimension or buffer layout differs across stages
- `unit-risk`: units are inferred or can drift
- `order-risk`: axis or ndarray order can drift
- `rename-needed`: current names hide materially different semantics
- `unknown`: current repository does not prove the intended contract

| Canonical field | Current CarbonPBS or `test_data` export | Current legacy CarbonPBS consumer | Current RTD compatibility consumer | Locked canonical contract for future `calDoseSubsecond` | Status |
| --- | --- | --- | --- | --- | --- |
| `rayweq_payload` | `rayweq`, `water_equivalence`; 1D float array with 9-value header plus body | `cudaCalDose.cpp` reads header as `[depth0, depthStep, nStep, y0, yStep, ny, x0, xStep, nx]`, uploads body as 3D texture with `width=nStep`, `height=ny`, `depth=nx` | `build_carbonpbs_context(...)` copies full payload into `waterEquivalence`, derives `raySpacing`, `spotDelta`, `steps`; wrapper may fill BEV directly from volume | Keep header plus payload as one explicit object. Units and axis order must be declared together with the header. No future code may treat `rayweq` as body-only. | `consistent` |
| `roi_linear_indices` | `roiIdx` exported as xyz triplets; `test_data` builds them from `np.unravel_index(...)` | Legacy CarbonPBS consumes triplets directly with `nRoi = size / 3` | Pybind accepts flat xyz triplets, `(N,3)`, or `(3,N)` and normalizes to `beamSettings.roiLinearIndices` | Canonical internal ROI representation is linear C-order indices over `(Nx,Ny,Nz)`. Boundary adapters may accept triplets but must normalize immediately. | `consistent` |
| `all_spot_energies` | `all_energies.shape = (6967,)` | Legacy CarbonPBS receives `sourceEne` as spot-wise input | Current pybind only checks length against `num_particles_per_beam`; wrapper does not preserve it as a first-class per-spot field | Future contract must either preserve spot-wise energies explicitly or declare them unused. Current repository cannot claim parity while they are only length-checked. | `ambiguous` |
| `source_position_world` | `sourcePos` in `dosecal.py`; fixture exports `source_pos.csv` | Legacy CarbonPBS final-dose-style paths read the first `vec3f` only | Pybind also keeps only the first `float3` via `first_float3_from_columns(source)` | Canonical contract requires one explicit world-space source point per beam. If spot-wise sources are materially different, they must be preserved under a separate field instead of being collapsed. | `ambiguous` |
| `spot_beam_directions_world` | `tmpBeamDir`, `beam_dir`; fixture stores spot-wise beam directions | Legacy CarbonPBS copies per-layer slices of beam directions to device | Pybind stores `spotBeamDirections` and also derives mean `beamDirection` | Canonical contract may keep both a spot-wise direction list and a nominal beam axis, but their roles must be explicit and separate. | `ambiguous` |
| `beam_axis_world` | Not exported independently; inferred from `tmpBeamDir` and basis vectors | Legacy CarbonPBS effectively uses beam directions directly | Wrapper computes nominal `beamDirection` as mean direction and uses it to sign-correct `bmZ` | Canonical contract requires one normalized nominal beam axis. If derived, the derivation rule must be explicit at the normalization boundary. | `ambiguous` |
| `beam_x_world` | `bmxdir` | Legacy CarbonPBS treats it as the beam x basis | Wrapper stores as `bmXDirection`, projects it against `bmZ`, and uses it as primary beam-plane x basis | Canonical contract requires a normalized world-space beam-plane x basis vector. | `consistent` |
| `beam_y_world` | `bmydir` | Legacy CarbonPBS treats it as the beam y basis | Wrapper stores as `bmYDirection`, uses it with `bmxdir` to build `bmZ`, and flips sign when building RTD gantry basis | Canonical contract requires a normalized world-space beam-plane y basis vector. | `consistent` |
| `sad` | scalar `sad` in `dosecal.py` and fixture | Legacy CarbonPBS forwards it into kernels as source-axis distance | Pybind stores `sad` and initializes `sourceDist = (sad, sad)` | Canonical contract keeps `sad` as scalar source-to-reference-plane distance in the same physical unit as geometry metadata. | `unit-risk` |
| `reference_plane` | Not exported as a dedicated field in `test_data`; `beamParaPos` comment references isocenter-relative position | Legacy CarbonPBS passes `beamParaPos` but does not expose a separate explicit reference-plane object | Wrapper hardcodes `refPlaneZ = 0.0f` and places the isocenter at `source + bmZ * sad` | Future contract must make the reference plane explicit. Current repository does not prove a dedicated exported variable for it. | `unknown` |
| `dose_grid_meta` | `doseGrid_corner`, `doseGrid_resolution`, `doseGrid_dims`; fixture pulls from `calc_required_meta.csv` | Legacy CarbonPBS copies these into `Grid.corner`, `Grid.resolution`, `Grid.dims` | Pybind passes them through, validates dims length, and uses them for output size and dummy CT allocation | Canonical contract keeps metadata in `(x,y,z)` order and forbids implicit ndarray-order inference. | `consistent` |
| `ct_grid_meta` | Not exported separately by the CarbonPBS compatibility fixture | Legacy CarbonPBS uses one `Grid` object in the compatibility path | Pybind compatibility path creates a dummy CT with the same dims as the dose grid | Future contract must distinguish `ctGrid` from `doseGrid`. Current compatibility path cannot prove a separate CT-grid contract. | `unknown` |
| `spot_weights` | `number_particle`, derived from `npermu * weight_vector / nFrac` | Legacy `cudaFinalDose(...)` consumes per-spot weights; legacy `cudaCalDose3(...)` does not expose this parameter | Current `calcDose`, `cuFinalDose`, and current `cuCalDose3` compatibility wrapper require and store `spotWeights` | Canonical contract requires explicit spot weights per spot. Any API that needs them must declare them rather than rely on legacy signature compatibility. | `shape-risk` |
| `layer_spot_counts` | `layerInfo.shape = (29,)` | Legacy CarbonPBS uses for layer loop bounds and beam offsets | Pybind stores as `layerSpotCounts`, validates total spot count, derives layer cutoffs | Canonical contract keeps one count per actual layer. | `consistent` |
| `layer_energy` | `layerEnergy.shape = (29,)` | Legacy CarbonPBS uses each value to select a LUT row through `binarySearchEneIdx(...)` | Pybind stores as actual layer energies; wrapper interpolates against LUT energy axes | Canonical contract keeps `layer_energy` as the actual plan-layer energy axis. | `consistent` |
| `energy_list` | `enelist.shape = (123,)` | Legacy CarbonPBS uses as machine LUT energy axis | Pybind uses it as `energyData.energiesPerU` and `profileEnergies` | Canonical contract keeps `energy_list` as a distinct LUT axis and forbids silent substitution for `layer_energy`. | `consistent` |
| `idbeamxy` | `idbeamxy.shape = (6967,2)`; fixture metadata says index-like WEQ lattice coordinate | Legacy CarbonPBS treats it as position on the `rayweq` departure lattice | Wrapper stores as `spotPositionsAreIndices = true`, then decodes using `raw`, `raw-0.5`, and `floor(raw)` variants in different code paths | Canonical contract must declare whether positions are WEQ indices or physical coordinates. If index-like, one decode formula must be locked and used everywhere. | `ambiguous` |
| `longitudinal_cutoff_per_spot` | `longitudalCutoff.shape = (6967,)` | Legacy CarbonPBS compares `weqDepth < longitudalCutoff` directly in kernels | Pybind collapses to one value per layer using the first spot in each layer; wrapper later maps it into RTD peak-depth logic | Future contract must either preserve spot-wise cutoff semantics or explicitly declare a layer-wise reduction rule. Current repository does not prove that layer-first collapse is lossless. | `ambiguous` |
| `transverse_cutoff` | `cutoff` scalar in `dosecal.py` call | Legacy CarbonPBS uses `transCutoff` as Gaussian-weight threshold | Current pybind compatibility accepts `cutoff` but does not use it | Future contract must either remove the field from the RTD compatibility API or reintroduce explicit semantics. Current behavior is not parity with legacy CarbonPBS. | `rename-needed` |
| `idd_table` | `idddata.shape = (123,4000)` plus `iddsetting` | Legacy CarbonPBS uploads differential IDD as 2D texture and indexes directly | Pybind converts differential IDD to CIDD using `iddsetting` and later fills missing tissue LUTs from `/tables` | Canonical contract must declare that `idddata` carries a depth axis plus explicit unit metadata. Silent mm or cm inference remains invalid. | `unit-risk` |
| `profile_table` | `profiledata.shape = (123,800,11)` plus `profilesetting` | Legacy CarbonPBS uploads 3D texture with `(channel, depth, energyRow)` access semantics | Pybind flattens and wrapper infers `profileRows`, `profileDepthN`, and `profileChannels` by shape and side information | Canonical contract must declare profile row axis explicitly and must not rely on shape inference alone. | `shape-risk` |
| `beam_para_table` | `beamparadata.shape = (123,3)` | Legacy CarbonPBS reads `(r2, rtheta, theta2)` by LUT energy row | Wrapper interpolates the same coefficients by `profileRowIdx` when profile override is enabled | Canonical contract must declare that row axis explicitly follows `energy_list` unless normalized. | `shape-risk` |
| `subspot_data` | `subspot_data.shape = (123,1,5)` | Legacy CarbonPBS uploads as 3D texture `(energyRow, subspot, channel)` with 5 channels | Pybind requires 3D shape `(rows, maxSubspots, 5)` and remaps to actual layer energies if needed; wrapper later uses the flattened data to drive exact RTD convolution or legacy CPB fallback | Canonical contract keeps five channels but requires explicit row-axis metadata and explicit normalization before wrapper consumption. | `shape-risk` |
| `beam_para_pos` | `beamParaPos` scalar | Legacy CarbonPBS forwards it with profile parameters | Wrapper keeps it only for profile override branches | Canonical contract keeps it as optional profile-model metadata. If profile override is not enabled, the field remains auxiliary. | `ambiguous` |
| `tables_dir` | implicit external dependency, default `tables/` | Not part of original CarbonPBS call signature | Current pybind loads density, stopping-power, radiation-length, and reference peak depths from `/tables` | Canonical contract must treat `/tables` as part of the RTD integration dependency graph, not as invisible ambient state. | `consistent` |

## `subspotData` Texture Contract

### Current observed layout

Legacy CarbonPBS and the current repository agree on one important fact: `subspotData` has five channels with stable semantics.

| Axis or channel | Legacy CarbonPBS meaning | Current RTD compatibility meaning | Locked contract |
| --- | --- | --- | --- |
| tensor shape | `(row_count, max_subspots_per_row, 5)` | Same storage shape after pybind coercion | Keep 3D tensor with explicit row axis and 5 channels |
| axis 0 | `eneIdx` row chosen from `energy_list` | Either direct row use or remapped row against `layer_energy` | Row axis must be declared as `energy_list`, `layer_energy`, or `unknown` |
| axis 1 | `isubspot` | same | keep as subspot index |
| axis 2 | channel | same | keep as 5 fixed channels |
| channel 0 | `deltaX` | used as subspot x offset in wrapper exact convolution | keep semantic fixed |
| channel 1 | `deltaY` | used as subspot y offset in wrapper exact convolution | keep semantic fixed |
| channel 2 | `weight` | used as subspot weight multiplier | keep semantic fixed |
| channel 3 | `sigmaX` | used as x sigma for exact RTD convolution | keep semantic fixed |
| channel 4 | `sigmaY` | used as y sigma for exact RTD convolution | keep semantic fixed |

### Current mapping into RayTraceDicom integration

Current wrapper behavior when the exact RTD path is available:

1. Reconstruct a dense raw spot lattice from `idbeamxy` and `spotWeights`.
2. For each layer, copy the dense spot plane.
3. For each non-zero subspot:
   - scale the dense spot plane by subspot weight
   - shift the spot plane by `deltaX` or `deltaY`
   - compute entry sigma from subspot sigma and air-scattering coefficients
   - call `performExactRTDConvolution2D(...)`
4. Accumulate all subspots into the ray-weight plane for that layer.

This means the locked contract between CarbonPBS and RTD is not merely "five channels exist". It also requires explicit alignment between:

- the row axis of `subspotData`
- the actual `layer_energy` axis used for dose accumulation
- the spot lattice reconstructed from `idbeamxy`

### Locked rule for future `calDoseSubsecond`

Future `calDoseSubsecond` input normalization must obey all of the following:

- `subspot_data` SHALL always be provided as `[row][subspot][channel=5]`.
- The provider SHALL declare whether `row` follows `layer_energy` or `energy_list`.
- If `row` follows `energy_list` instead of `layer_energy`, normalization SHALL remap explicitly and record that remap.
- If the provider cannot prove the row axis, the field SHALL be marked `unknown` and SHALL not be silently inferred from row count alone.

## ROI Contract

### Current observed input forms

Current pybind compatibility accepts:

- flat 1D xyz triplets
- 2D `(N,3)`
- 2D `(3,N)`

It converts all of them to linear indices with:

`linear = ix * (Ny * Nz) + iy * Nz + iz`

### Locked canonical ROI contract

| Aspect | Locked rule |
| --- | --- |
| boundary compatibility | Python adapters may accept flat xyz triplets, `(N,3)`, or `(3,N)` |
| canonical internal form | `roi_linear_indices` only |
| linearization order | C-order over `(Nx,Ny,Nz)` |
| relation to `vec3i ROI` | `vec3i` forms are boundary-only and documentation-only once normalization finishes |
| failure rule | Any shape outside the accepted boundary forms is rejected; any ambiguous grid order is rejected or marked `unknown` before normalization |

### Why this is required

The current repository already contains:

- legacy CarbonPBS logic that still thinks in xyz triplets
- pybind normalization that converts to linear indices
- wrapper logic that only uses linear indices

Keeping more than one internal ROI representation alive in `calDoseSubsecond` would recreate the current ambiguity.

## Beam Geometry And Coordinate-System Contract

| Canonical geometry field | CarbonPBS or `test_data` evidence | Current RTD wrapper field or derived quantity | Locked contract | Status |
| --- | --- | --- | --- | --- |
| `beam_x_world` | `bmxdir` from `dosecal.py` and fixture | `bmXDirection`, later orthogonalized into `bmX` | normalized world-space beam-plane x basis vector | `consistent` |
| `beam_y_world` | `bmydir` from `dosecal.py` and fixture | `bmYDirection`, later orthogonalized into `bmY` | normalized world-space beam-plane y basis vector | `consistent` |
| `beam_axis_world` | inferred from spot `beam_dir` and `cross(bmxdir,bmydir)` | wrapper sign-corrected `bmZ` and gantry `gZ=-bmZ` | one explicit normalized nominal beam axis; if derived from basis vectors, derivation must be documented | `ambiguous` |
| `spot_beam_directions_world` | spot-wise `tmpBeamDir` export | `spotBeamDirections` and virtual-source estimation | optional spot-wise direction list with explicit role | `ambiguous` |
| `source_world` | `sourcePos` export | `sourcePosition` using the first `float3` only | one explicit world-space source point per beam | `ambiguous` |
| `sad` | scalar export from `dosecal.py` and fixture | `sad`, then `sourceDist` initialization | distance from source to reference plane along beam axis | `unit-risk` |
| `reference_plane` | not explicitly exported as its own field | wrapper sets gantry origin at `iso = source + bmZ * sad`, `refPlaneZ = 0` | explicit plane perpendicular to beam axis; must be exported or derivable without silent inference | `unknown` |
| `gantry_basis` | not exported directly | wrapper uses `gX=bmX`, `gY=-bmY`, `gZ=-bmZ` | internal RTD-only basis derived from explicit world-space beam geometry | `consistent` |

Locked geometry rules for future `calDoseSubsecond`:

- `bmxdir` and `bmydir` are the primary beam-plane basis inputs.
- `beamDirection` or equivalent nominal beam axis must be explicitly normalized and must be consistent with `cross(bmxdir,bmydir)`; otherwise normalization fails.
- `source` and `sad` must define the reference plane explicitly instead of relying on hidden wrapper assumptions.
- Any per-spot source or per-spot direction field must be explicitly marked as per-spot rather than silently collapsed to the first element or the mean.

## `doseGrid` And `ctGrid` Metadata Versus Python Numpy Shape

### Current observed metadata order

Across `dosecal.py`, `test_data`, and the CarbonPBS compatibility layer, metadata vectors are expressed in `(x, y, z)` order:

- `doseGrid_dims = (Nx, Ny, Nz)`
- `doseGrid_corner = (x0, y0, z0)`
- `doseGrid_resolution = (dx, dy, dz)`

### Current observed ndarray order drift

| Surface | Current ndarray order | Evidence | Risk |
| --- | --- | --- | --- |
| `dosecal.py` `finalDose` | `(Nx, Ny, Nz)` | `np.zeros((self.doseGrid.dims[0], self.doseGrid.dims[1], self.doseGrid.dims[2]))` | CarbonPBS-compatible output buffer appears XYZ-shaped |
| `test_data` `finalDose` | `(Nx, Ny, Nz)` | `np.zeros(tuple(int(v) for v in dose_grid_dims))` | Fixture follows XYZ-shaped buffer |
| direct `raytracedicom_wrapper_py(...)` return | `(Nz, Ny, Nx)` | pybind doc says `shape (Z, Y, X)` | Direct RTD API uses ZYX-shaped buffer |
| current `cuCalDose3` compatibility `dose_grid` return | `(Nz, Ny, Nx)` | pybind creates `py::array_t<float> dose_arr({dims[2], dims[1], dims[0]})` | Compatibility wrapper return order differs from CarbonPBS fixture order |
| current `calcDose` or `cuFinalDose` in-place output | size-checked only | pybind validates total size, not semantic axis order | Buffer-order bugs can pass silently if total element count matches |

### Locked grid contract

| Aspect | Locked rule |
| --- | --- |
| metadata vector order | always `(x, y, z)` |
| ndarray order | must be declared explicitly at the boundary; implicit order is forbidden |
| canonical internal contract | metadata and buffers are normalized once before execution; future `calDoseSubsecond` code must not mix XYZ and ZYX buffers without an explicit adapter |
| `ctGrid` vs `doseGrid` | must be separate metadata objects in the canonical contract |
| failure rule | if a caller supplies metadata and a buffer shape without declaring order, normalization must reject or mark `unknown` |

## `dosecal.py` And `test_data` Consistency Matrix

| Exported field | `dosecal.py` or `test_data` meaning | Actual legacy CarbonPBS use | Actual RTD-wrapper use | Result |
| --- | --- | --- | --- | --- |
| `rayweq` | WEQ header plus payload | consumed directly | consumed directly | `consistent` |
| `roiIdx` | xyz ROI triplets | consumed directly | normalized to linear indices | `consistent` after normalization |
| `all_energies` | per-spot energies | consumed as spot-wise energy input | length-checked only | `ambiguous` |
| `sourcePos` | source position export | first source used | first source used | `ambiguous` if spot-wise source data matters |
| `tmpBeamDir` | per-spot directions | consumed directly | stored and averaged | `ambiguous` |
| `bmxdir` / `bmydir` | beam-plane basis | consumed directly | consumed directly | `consistent` |
| `doseGrid_corner` / `resolution` / `dims` | XYZ metadata | consumed directly | consumed directly, but ndarray order still drifts | `order-risk` |
| `longitudalCutoff` | per-spot cutoff | compared directly to WEQ depth | reduced to layer value then remapped into RTD peak-depth logic | `ambiguous` |
| `enelist` | LUT energy axis | consumed directly | consumed directly | `consistent` |
| `idddata` / `iddsetting` | IDD tensor plus depth axis | consumed directly | rebuilt into CIDD with unit heuristics | `unit-risk` |
| `profiledata` / `profilesetting` | profile tensor plus depth axis | consumed directly | inferred by wrapper shape logic | `shape-risk` |
| `beamparadata` | `(r2, rtheta, theta2)` LUT rows | consumed directly | consumed directly, but row axis may differ from actual layers | `shape-risk` |
| `subspotdata` | five-channel subspot tensor on LUT axis | consumed directly | may be remapped to layer axis | `shape-risk` |
| `layerInfo` / `layerEnergy` | actual plan-layer counts and energies | consumed directly | consumed directly | `consistent` |
| `idbeamxy` | WEQ-lattice index-like coordinate | consumed directly | decoded inconsistently across wrapper paths | `ambiguous` |
| `number_particle` | spot weights | used by final-dose path | required by current RTD compatibility, including `cuCalDose3` | `shape-risk` due to legacy signature drift |
| `cutoff` | lateral weight threshold | consumed directly | ignored | `rename-needed` |
| `beamParaPos` | profile-model helper scalar | consumed directly | auxiliary RTD profile override field | `ambiguous` |
| `/tables` | not an exported dosecal field | not part of original CarbonPBS call contract | required by current RTD compatibility for density/SP/RRL and reference peak depths | `consistent` within RTD path, but external dependency must be explicit |

## Checklist

### Already consistent enough to lock

- `rayweq` header structure and payload role
- `layerInfo` as one count per actual layer
- `layerEnergy` as actual plan-layer energy axis
- `energy_list` as distinct LUT energy axis
- `bmxdir` and `bmydir` as primary beam-plane basis vectors
- the five `subspotData` channel meanings
- internal ROI linearization formula once normalization is complete

### Ambiguous and must not be silently inferred

- whether `all_energies` is required semantically or only as legacy compatibility baggage
- whether per-spot `sourcePos` variation is meaningful or can be collapsed to one source
- whether `beamDirection` is a true independent input or only a sign-disambiguation helper
- whether `longitudal_cutoff` is allowed to vary inside a layer
- whether `beamParaPos` is required for any non-profile branch
- the exact exported reference-plane definition

### Rename or shape changes needed in future implementation

- `cutoff` should not remain a live argument name if the RTD compatibility path ignores it
- `subspot_data`, `profile_data`, and `beam_para_data` need explicit row-axis metadata
- `cuCalDose3` compatibility should not hide the fact that it needs spot weights while the legacy signature did not
- `ctGrid` and `doseGrid` must be distinct canonical objects instead of a dummy alias

### Unit or coordinate-order risks

- `lenToMm`, `iddDepthUnitToMm`, and `energyDepthToMm` heuristics
- `idbeamxy` decode drift between `raw`, `raw-0.5`, and `floor(raw)`
- implicit assumption that metadata `(Nx,Ny,Nz)` uniquely determines ndarray order
- mixed use of XYZ-shaped and ZYX-shaped Python buffers
- potential mismatch between exported grid units and `/tables` depth units

### Fields that remain `unknown` until explicitly supplied

- independent exported reference-plane metadata
- independent `ctGrid` contract for the CarbonPBS compatibility path
- any field whose unit cannot be proven from source or fixture metadata alone

## High-Risk Misalignment Points

1. `idbeamxy` is explicitly exported as a WEQ-lattice index-like coordinate, but the wrapper decodes it differently in spot-bounds, source-distance, and dense-lattice reconstruction paths.
2. `subspot_data`, `profile_data`, and `beam_para_data` are exported on a 123-row LUT axis while actual plan layers are 29 rows; the current repository does not normalize all of them in one unified way.
3. `cutoff` is active in legacy CarbonPBS kernels and ignored in the current RTD compatibility path.
4. `all_energies` is exported as a spot-wise field but is currently only length-checked in the RTD compatibility path.
5. The current repository mixes XYZ metadata with both XYZ and ZYX ndarray orders, which means shape-compatible buffers can still be semantically wrong.
6. The current compatibility path aliases CT metadata to dose-grid metadata by creating a dummy CT, so any future implementation that needs a real CT contract must make that distinction explicit first.

## Future `calDoseSubsecond` Input Contract

The future `calDoseSubsecond` implementation must normalize to the following canonical contract before execution. This is the execution gate that this change is designed to create.

| Canonical input | Required form | Required metadata | Normalization rule | If not provable |
| --- | --- | --- | --- | --- |
| `rayweq_payload` | header plus body | explicit axis order and units | keep together as one object | reject or mark `unknown` |
| `roi_linear_indices` | 1D linear integer array | `doseGrid.dims = (Nx,Ny,Nz)` | normalize from accepted Python forms at boundary only | reject |
| `dose_grid_meta` | `dims`, `corner`, `resolution` in `(x,y,z)` | explicit ndarray order for every buffer | normalize once and record order | reject or mark `unknown` |
| `ct_grid_meta` | separate from `dose_grid_meta` | explicit `(x,y,z)` metadata and buffer order | do not alias silently to dose grid | mark `unknown` if unavailable |
| `beam_geometry` | `source`, `sad`, `bmxdir`, `bmydir`, nominal beam axis | explicit world frame and normalized vectors | validate basis consistency and reference-plane rule before execution | reject or mark `unknown` |
| `spot_positions` | one explicit semantic mode | `spot_positions_kind = rayweq_index` or `physical_reference_plane` | decode exactly once according to declared mode | reject |
| `spot_weights` | one weight per spot | total spot count must match `layerInfo` | normalize once and preserve explicitly | reject |
| `layer_energy` and `layerInfo` | actual plan-layer axes | same row count | preserve directly | reject |
| `subspot_data` | `[row][subspot][5]` | explicit row-axis declaration | remap explicitly if row axis is not `layer_energy` | reject or mark `unknown` |
| `profile_data` and `beam_para_data` | explicit row-aligned model tensors | explicit row axis, depth axis, units | normalize explicitly before wrapper consumption | reject or mark `unknown` |
| `idd_data` | differential or cumulative status explicit | depth axis and units explicit | convert at boundary with no silent unit guess | reject or mark `unknown` |

Implementation rule:

- future `calDoseSubsecond` code SHALL not rely on silent row-count inference, silent unit heuristics, or silent ndarray-order inference for any normalized input field that can be declared explicitly

## Risks / Trade-offs

- [Risk] Locking the contract will surface existing inconsistencies that current code tolerates implicitly. -> Mitigation: classify each drift precisely and separate boundary compatibility from canonical internal representation.
- [Risk] Some fields cannot be proven from current source alone, especially reference-plane and CT-grid semantics. -> Mitigation: mark them `unknown` and require future code to reject or annotate them explicitly.
- [Risk] Forcing explicit ndarray order and row-axis metadata adds adapter work. -> Mitigation: it removes the silent shape-compatible failures already present in the current chain.
- [Risk] Keeping algorithm behavior unchanged means the change cannot solve every runtime mismatch immediately. -> Mitigation: the checklist makes those mismatches explicit and turns them into follow-up implementation tasks.

## Migration Plan

1. Accept this change as the source of truth for the integration contract.
2. Normalize existing documentation, adapter code, and fixtures to the canonical field names and metadata requirements.
3. Add boundary validation that rejects or flags any field whose status in this design is ambiguous or `unknown`.
4. Implement `calDoseSubsecond` only after its input builder can produce the canonical contract defined here.

Rollback strategy:

- If the contract-locking direction is rejected, remove the OpenSpec change only. No algorithm code changes are proposed in this stage.

## Open Questions

- Is the future boundary contract allowed to keep XYZ-shaped CarbonPBS output buffers while the direct RTD API keeps ZYX-shaped returns, or must both be unified behind one public Python order?
- Can `longitudal_cutoff` legally vary within one layer in real exports, or must that become a validation error?
- Should `all_energies` remain part of the future canonical contract, or should the contract explicitly drop it from the RTD path if spot-wise energy is not needed?
- What explicit field should carry reference-plane semantics if `beamParaPos` is insufficient or profile-specific?
