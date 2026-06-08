## Context

This change repurposes `carbonpbs-contract-audit` from an implementation proposal into an audit dossier. The objective is to freeze the current CarbonPBS-to-RTD contract before any code changes, because the repository currently contains three materially different contract layers:

- the original CarbonPBS implementation in `carbonPBS/cudaCalDose.cpp` and `carbonPBS/deviceCalDose.cu`
- the current compatibility adapter in `src/bindings/raytracedicom_pybind.cpp`
- the heavily diverged RTD execution path in `src/core/raytracedicom_wrapper.cu` and `src/algorithms/convolution.cu`

Source basis reviewed for this audit:

- `carbonPBS/cudaCalDose.cpp`
- `carbonPBS/deviceCalDose.cu`
- `src/core/raytracedicom_wrapper.cu`
- `src/algorithms/convolution.cu`
- `src/bindings/raytracedicom_pybind.cpp`
- `include/core/raytracedicom_integration.h`
- `src/tests/carbonpbs_data_layout_example.cu`
- `src/tests/wrapper_integration_test.cu`
- `tps_py/dosecal.py`
- `test_data/rtd_test_case_bg800_beam925.py`
- `test_data/dose_inputs_csv/variable_name_meta.csv`
- `test_data/dose_inputs_csv/calc_required_meta.csv`
- `tables/proton_cumul_ddd_data.txt`
- `tables/density_Schneider2000_adj.txt`
- `tables/HU_to_SP_H&N_adj.txt`
- `tables/radiation_length.txt`
- `tables/generated/README.md`
- `tables/generated/proton_from_rtd_for_carbonpbs_info.json`
- upstream reference files in `RayTraceDicom-main (1)/RayTraceDicom-main/src/`

Observed `bg800_beam925` fixture facts used below:

- `layer_energy`: `(29,)`
- `layer_info`: `(29,)`
- `all_energies`: `(6967,)`
- `number_particle`: `(6967,)`
- `longitudal_cutoff`: `(6967,)`
- `energy_list`: `(123,)`
- `idd_data`: `(123, 4000)`
- `profile_data`: `(123, 800, 11)`
- `beam_para_data`: `(123, 3)`
- `subspot_data`: `(123, 1, 5)`
- `idbeamxy`: `(6967, 2)`
- `calc_required_meta.csv` declares `rayweq_n_step=4455`, `rayweq_y0=-50`, `rayweq_dy=1`, `rayweq_ny=101`, `rayweq_x0=-52`, `rayweq_dx=1`, `rayweq_nx=105`
- `calc_required_meta.csv` explicitly states: `idbeamxy is rayweq texture index-like coordinate, not direct gantry-plane physical coordinate`

## Goals / Non-Goals

**Goals:**

- Record the live variable contract across legacy CarbonPBS, the current pybind adapter, and the current RTD wrapper.
- Record the live coordinate and texture contracts, including places where different parts of the code disagree.
- Trace `dosecal.py` and `test_data` fields to their actual consumers, including fields that are ignored or only used in tests.
- Enumerate the current repository's material deviations from `RayTracedicom_main`.
- Rank the most likely Bragg peak position error causes from the current contract rather than from intended behavior.

**Non-Goals:**

- Changing any runtime code, bindings, tests, or LUT contents.
- Proving the final numerical root cause by execution; this dossier is a static source audit.
- Rewriting the CarbonPBS or RTD APIs in this change.
- Deciding implementation fixes; that belongs to follow-up work after this audit is accepted.

## Decisions

### Decision: Audit actual current behavior, not intended exporter semantics

This dossier treats the checked-in source as authoritative even when comments, fixture notes, and upstream assumptions disagree. If a value is documented one way but consumed another way, the consumed behavior is recorded as the contract.

Rationale:

- The current bug risk comes from runtime behavior, not from intended design.
- The repository already contains multiple partially overlapping descriptions of the same inputs.
- Implementation work needs one source-anchored baseline before any refactor.

Alternatives considered:

- Document only the intended CarbonPBS exporter semantics. Rejected because it hides the existing drift that is most likely causing the current errors.

### Decision: Keep legacy CarbonPBS and current RTD compatibility paths side by side

The audit compares the original CarbonPBS path and the current RTD compatibility path rather than collapsing them into one blended description.

Rationale:

- The user explicitly requested deviations from `RayTracedicom_main`.
- Several high-risk mismatches only appear when comparing the three layers together: original CarbonPBS, current pybind adapter, and current RTD wrapper.
- `carbonPBS/cudaCalDose.cpp` still defines the original call and texture contracts that `dosecal.py` was written against.

Alternatives considered:

- Audit only `src/core/raytracedicom_wrapper.cu`. Rejected because the binding layer and legacy module change the meaning of inputs before the wrapper sees them.

### Decision: Treat `test_data` and `/tables` as part of the live contract

The fixture CSVs and table readers are included in the audit as first-class contract evidence, not as auxiliary documentation.

Rationale:

- The current pybind path builds its energy struct partly from `idddata` and partly from `/tables`.
- The fixture metadata explicitly documents several semantics that are easy to lose in code, especially `idbeamxy`.
- The generated-table docs expose the current reference LUT ranges and therefore help explain energy-axis clamp risks.

Alternatives considered:

- Limit the audit to runtime code only. Rejected because that would miss the fixture-side semantics and the actual `/tables` dependencies.

## Variable Contract Table

The table below records the live contract for the major CarbonPBS compatibility surfaces. The `Observed` column uses the `bg800_beam925` fixture when available.

| Surface | Observed or expected shape | Legacy CarbonPBS contract | Current RTD compatibility contract | Primary consumers and audit notes |
| --- | --- | --- | --- | --- |
| `rayweq` / `weq_data` / `water_equivalence` | 1D float array; first 9 values are header, payload is ray lattice body | `carbonPBS/cudaCalDose.cpp` treats first 9 floats as `[depth0, depthStep, nStep, y0, yStep, ny, x0, xStep, nx]` and uploads payload as texture with `width=nStep`, `height=ny`, `depth=nx` | `build_carbonpbs_context(...)` copies the entire array into `beamSettings.waterEquivalence`, copies header into `rayWeqHeader`, sets `raySpacing=(header[7], header[4])`, `spotDelta=(header[7], header[4], 0)`, and `steps=round(header[2])`; wrapper may bypass CT tracing and fill BEV directly from this volume | Legacy consumers: `cudaFinalDose`, `cudaCalDose3`, device kernels. Current consumers: `src/bindings/raytracedicom_pybind.cpp`, `src/core/raytracedicom_wrapper.cu::fillBevFromWeqVolumeKernel`. This is the authoritative geometry source for `idbeamxy` decoding in the current path. |
| `roiIdx` / `roi_index` | Either flattened xyz triplets or 2D `(N,3)` or `(3,N)` int array | Legacy CarbonPBS uses `nRoi = roiIndex.size / 3` and consumes xyz triplets directly | Current pybind converts triplets to linear dose-grid indices `x*(Ny*Nz)+y*Nz+z`; wrapper stores them in `beamSettings.roiLinearIndices` | Current wrapper uses ROI indices for bounds analysis and optional ROI-derived water phantom in tests, but CPB coverage still uses the full projected dose volume. |
| `all_energies` / `source_energies` | `(6967,)` float32 per-spot energies | Legacy CarbonPBS uses `sourceEne` as per-spot or per-beam energy input and copies layer-local beam directions with the same spot cardinality | Current pybind reads `source_energies` only to verify length against `num_particles_per_beam`; it does not store these per-spot energies in `RTDBeamSettings` and instead uses `layer_energy` as the effective beam energy list | This is a live contract loss: per-spot energies are not propagated into the current wrapper path. |
| `sourcePos` / `source` | Exported as per-spot or beam-aligned `N x 3` float array | Legacy CarbonPBS reads only the first `vec3f` as `sourcePos` for final-dose-style paths | Current pybind also takes only the first `float3` via `first_float3_from_columns(source)` and stores it as a single global `sourcePosition` | Any per-spot source variation in the exporter is silently collapsed to one source in both paths. |
| `tmpBeamDir` / `beam_dir` | `(6967, 3)` or flattened spot directions | Legacy CarbonPBS copies per-layer slices of spot beam directions to device and uses them during layer computation | Current pybind stores all spot beam directions in `beamSettings.spotBeamDirections` and also derives one mean `beamDirection` via `normalized_mean_direction(...)` | The current wrapper uses spot directions both for mean beam-direction inference and virtual-source estimation. |
| `bmxdir` | `(3,)` | Legacy CarbonPBS treats it as the exported beam-x basis vector | Current pybind stores it as `beamSettings.bmXDirection`; wrapper uses it to construct `bmX`, `bmZ`, and gantry basis `gX` | This basis is now part of the active geometry contract, not just metadata. |
| `bmydir` | `(3,)` | Legacy CarbonPBS treats it as the exported beam-y basis vector | Current pybind stores it as `beamSettings.bmYDirection`; wrapper uses it to construct `bmY`, `bmZ`, and gantry basis `gY` | `gY = -bmY` in the wrapper, which is an explicit sign convention not present in upstream `BeamSettings`. |
| `doseGrid.corner` / `corner` | `(3,)` | Legacy CarbonPBS copies into `Grid.corner` | Current pybind passes through to wrapper dose geometry and dummy CT geometry | The current compatibility path uses this dose geometry together with imported WEQ; there is no real CT grid passed through the CarbonPBS compatibility API. |
| `doseGrid.resolution` / `resolution` | `(3,)`; fixture is `(2,2,2)` | Legacy CarbonPBS copies into `Grid.resolution` | Current pybind passes through; wrapper uses `ctResolution.z` to infer `lenToMm` and therefore to decide whether to scale cm-like inputs to mm | `lenToMm = 10` only when `ctResolution.z < 0.3`, otherwise `1`. This heuristic is a live unit contract. |
| `doseGrid.dims` / `dims` | `(3,)` ints; fixture is `(211,152,217)` | Legacy CarbonPBS copies into `Grid.dims` | Current pybind validates exactly 3 integers, uses them for dummy CT allocation, output-size validation, and ROI linearization | The compatibility path uses dose dims as both CT dims and dose dims when building the dummy CT. |
| `longitudalCutoff` / `longitudal_cutoff` | `(6967,)` per-spot float array; fixture range is approximately `239.1` to `460.9` | Legacy CarbonPBS reads `longitudalCutoff[beamOffset]` for each layer and device kernels compare `weqDepth < longitudalCutoff` directly | Current pybind requires length equal to total spot count, then collapses to one value per layer using the first spot in each layer; wrapper converts that layer value to mm and then maps `cutoff -> peakDepth / BP_DEPTH_CUTOFF` | This is one of the most important contract changes in the repository because the current wrapper no longer uses the cutoff the same way as legacy CarbonPBS. |
| `enelist` / `energy_list` | `(123,)` float32 LUT energy axis | Legacy CarbonPBS uses `binarySearchEneIdx(layerEnergy[i], enelist, nEne)` to select LUT rows for `beamparadata`, `idddata`, `profiledata`, and `subspotData` | Current pybind stores `enelist` as both `energyData.energiesPerU` and `beam.profileEnergies`; wrapper later interpolates fractional energy indices against this axis | In the current path the same axis drives CIDD interpolation, profile row selection, and optional subspot remapping. |
| `idddata` | `(123,4000)` | Legacy CarbonPBS uploads as 2D texture with `width=nSamples`, `height=nEnergies` | Current pybind requires rows equal `enelist.length`, integrates differential IDD into cumulative CIDD using `iddsetting[1]`, and stores the result in `energyData.ciddMatrix` | Current pybind does not use exporter `export_mode` metadata to alter depth handling; it trusts array rows and `iddsetting`. |
| `iddsetting` | `(3,)`; fixture `[0.05, 0.1, 4000]` | Legacy CarbonPBS copies a `vec3f` `(start, step, count)` and uses it directly for depth indexing into `idddata` | Current pybind interprets it as `[start, step, nSamples]`, applies heuristic `iddDepthUnitToMm`, and sets `scaleFacts = 1 / step_mm` for the built CIDD | This is the direct WEPL-to-IDD index contract in the current path. |
| `profiledata` | `(123,800,11)` | Legacy CarbonPBS uploads as 3D texture with `width=nProfilePara`, `height=nProfileDepth`, `depth=nEne` | Current pybind flattens it, wrapper infers `profileRows`, `profileDepthN`, and `profileChannels`, then creates a 3D texture when shape inference succeeds | The wrapper does not consume explicit `export_mode` metadata; it infers shape from tensor size, `profileSetting`, `beamParaData`, `profileEnergies`, or `numLayers`. |
| `profilesetting` | `(3,)`; fixture `[0.25, 0.5, 800]` | Legacy CarbonPBS copies a `vec3f` `(depth0, depthStep, depthN)` | Current pybind passes it through to wrapper; wrapper uses it to determine profile depth axis and texture shape | `profilesetting` is not equivalent to `raySpacing`; the fixture metadata calls this out explicitly. |
| `beamparadata` / `beamParaData` | `(123,3)` | Legacy CarbonPBS indexes `beamparadata[energyIdx*3 + {0,1,2}]` as `(r2, rtheta, theta2)` | Current pybind flattens the whole array; wrapper later interpolates the same 3 coefficients by `profileRowIdx` when the Carbon profile override is enabled | The current path can use a row count different from actual plan layers because `profileRowIdx` may be evaluated on the LUT energy axis. |
| `subspotdata` / `subspot_data` | `(123,1,5)` in the fixture | Legacy CarbonPBS uploads as 3D texture with `width=5`, `height=nSubspot`, `depth=nEne`; channels are `deltaX, deltaY, weight, sigmaX, sigmaY` | Current pybind requires shape `(num_rows, max_subspots_per_layer, 5)`; if row count differs from actual layer count, it remaps or interpolates the LUT rows from `enelist` onto `layer_energy` | This is a major contract change because the current wrapper can execute with `subspot_data.shape[0] != layer_energy.size()`, whereas legacy CarbonPBS expects a direct energy-axis lookup. |
| `layerInfo` / `layer_info` | `(29,)` int32 spot count per layer | Legacy CarbonPBS uses it to determine layer sizes, beam offsets, maximum layer size, and per-layer loop bounds | Current pybind stores it in `beamSettings.layerSpotCounts` and uses it to validate total spot count and derive layer cutoffs | In the current path `layer_info` must match `layer_energy.size()`, but it does not have to match LUT row counts for `subspot_data`, `profiledata`, or `beamparadata`. |
| `layerEnergy` / `layer_energy` | `(29,)` float32 actual plan-layer energies | Legacy CarbonPBS uses each layer energy to choose a LUT row via `binarySearchEneIdx(...)` | Current pybind stores it as `beamSettings.energies`; wrapper uses it as the actual energy list, interpolating against `energyData.energiesPerU` and `beam.profileEnergies` | In the fixture the actual layer range (`242.95` to `359.44`) is much narrower than the LUT energy range (`120.26` to `399.92`). |
| `idbeamxy` | `(6967,2)` float32 | Legacy CarbonPBS uses it as per-spot position on the `rayweq` departure lattice | Current pybind stores it as `spotPositions` and forces `spotPositionsAreIndices = true`; wrapper decodes it using multiple formulas depending on the call site | This field is explicitly documented in fixture metadata as index-like, not physical. The wrapper currently uses `raw`, `raw-0.5`, and `floor(raw)` variants in different places. |
| `number_particle` / `num_particles_per_beam` / `npermu * weight_vector / nFrac` | `(6967,)` float32 per-spot weights in the fixture | Legacy `cudaFinalDose` uses per-spot weights; legacy `cudaCalDose3` does not accept this array in its signature | Current `calcDose` and `cuFinalDose` require it and store it as `beamSettings.spotWeights`; current `cuCalDose3` compatibility wrapper also requires it even though the legacy CarbonPBS `cuCalDose3` signature did not | This is a live signature deviation between the current pybind compatibility layer and legacy CarbonPBS. |
| `sad` | scalar float; fixture around `5802.7` | Legacy CarbonPBS uses it as source-axis distance | Current pybind stores it as `beamSettings.sad` and also initializes `sourceDist = (sad, sad)` | Wrapper later may infer source distance again from `spotBeamDirections` if explicit values are not trusted. |
| `cutoff` / `transCutoff` | scalar float; fixture passes `0.00005` in `cuFinalDose` path | Legacy device kernels use it as lateral Gaussian threshold via `if (gaussianWeight > transCutoff)` | Current pybind compatibility wrappers accept `cutoff` for signature compatibility but mark it unused and never pass it into the RTD wrapper | This field is live in legacy CarbonPBS and ignored in the current compatibility path. |
| `beamParaPos` / `beam_para_pos` | scalar float; fixture uses `0.0` | Legacy CarbonPBS forwards it into kernels alongside profile parameters | Current pybind stores it as `beamSettings.beamParaPos`; wrapper uses it only when the Carbon profile override path is enabled | This field remains wired through, but only one optional override path consumes it now. |
| `tables_dir` | string path; default is `tables/` | No direct legacy CarbonPBS equivalent in the original `dosecal.py -> carbonPBS` call | Current pybind uses it to call `energyReader(tables_dir)` and populate density, stopping-power, radiation-length, and reference `peakDepths` | `/tables` are therefore part of the live CarbonPBS compatibility contract in the current path even though they were not part of the original CarbonPBS call surface. |

Immediate audit observations from the variable contract:

- `all_energies` is validated but not propagated into the current wrapper as per-spot energy data.
- `cutoff` is active in legacy CarbonPBS and ignored in the current pybind compatibility path.
- `num_particles_per_beam` is newly required by the current `cuCalDose3` compatibility wrapper even though legacy `cudaCalDose3(...)` did not accept it.

## Coordinate-System Table

| Space | Definition in current repo | Origin and axes | Units | Current consumers | Audit notes |
| --- | --- | --- | --- | --- | --- |
| Dose-grid index space | Integer `(x,y,z)` voxel indices and C-order linearization `x*(Ny*Nz)+y*Nz+z` | Origin at voxel `(0,0,0)` of dose volume | index units | `to_roi_linear_indices(...)`, output validation, ROI handling | This is the canonical ROI indexing used by the current wrapper after pybind conversion. |
| Dose-grid world space | `corner + index * resolution` | Exported dose corner plus dose basis implied by `doseGrid` orientation already resolved in `dosecal.py` | exporter units; wrapper may reinterpret via `lenToMm` | Legacy CarbonPBS grid setup and current wrapper geometry setup | The compatibility API does not carry the full dose-grid orientation matrix; it receives already rotated beam basis vectors instead. |
| Dummy CT image space | Current CarbonPBS compatibility path allocates a synthetic CT array with the same dims as the dose grid and fills it with `1000.0f` | Same dims, resolution, and corner as the dose grid | same as dose grid | `make_dummy_ct_from_dims(...)`, `subsecondWrapper(...)` | This is a current-path-only construct. Upstream RTD expects actual CT data and affine transforms. |
| Exporter beam basis | `tmpBeamDir`, `bmxdir`, `bmydir`, `sourcePos`, `sad` exported by `dosecal.py` | `dosecal.py` computes beam basis in its own world frame before calling compatibility APIs | exporter units | Legacy CarbonPBS, current pybind, current wrapper | This is the only explicit beam/world transform data that survives into the current compatibility path. |
| WEQ header lattice space | `rayweq` header `[z0,dz,nStep, y0,dy,ny, x0,dx,nx]` defines the departure-plane lattice plus depth axis | X and Y lie on the departure plane; depth axis is stored as the fastest texture dimension in the payload | header units; current fixture metadata suggests mm-like values | Legacy texture creation, current ray-spacing derivation, current WEQ import | This space is the authoritative reference for `idbeamxy` in the fixture metadata. |
| Raw `idbeamxy` space | Exported as index-like coordinates on the WEQ lattice, typically `physical_index + offset + 0.5` in `dosecal.py` | Origin aligned to WEQ texel centers rather than plane boundary | WEQ-lattice index units | Legacy CarbonPBS, current wrapper spot reconstruction | The fixture metadata explicitly warns that these are not direct physical beam-plane coordinates. |
| Wrapper decoded spot physical space | Current wrapper decodes `idbeamxy` in three different ways depending on context | `decodeSpotPosition(...)` uses `x0 + raw*dx`; `estimateVirtualSourceDistancesFromSpots(...)` uses `x0 + (raw-0.5)*dx`; `buildRawSpotLattice(...)` uses `ix=floor(raw)` with `spotOffset=x0+0.5*dx` | header units, then sometimes divided by `lenToMm` | Spot bounds, source-distance inference, raw spot lattice reconstruction | This internal inconsistency is a live coordinate-contract hazard. |
| Wrapper beam-local basis (`bmX`, `bmY`, `bmZ`) | Constructed from `bmXDirection`, `bmYDirection`, and mean `beamDirection` | `bmZ = cross(bmXDirectionW, bmYDirectionW)` with sign corrected toward mean beam direction | normalized direction vectors | `src/core/raytracedicom_wrapper.cu` basis construction | This is not upstream `BeamSettings` behavior; upstream expects transforms to already be resolved. |
| RTD gantry space (`gX`, `gY`, `gZ`) | Wrapper world-to-gantry frame | `gX = bmX`, `gY = -bmY`, `gZ = -bmZ`; isocenter at `source + bmZ * SAD` and gantry `z=0` | mm after `lenToMm` normalization | Wrapper CPB, fan, and transfer transforms | The sign flip on `gY` and `gZ` is a deliberate RTD compatibility convention in the current wrapper. |
| CPB and ray plane space | 2D gantry-plane grid at the reference plane | `cpbCorner`, `cpbResolution`, and `rayDims` built either from raw spot lattice, decoded spot bounds, or WEQ header | wrapper internal base units before fan conversion; later converted to mm | Current convolution path, BEV fill, RTD superposition setup | The current wrapper can crop this plane by decoded spot bounds or derive it from the full WEQ header. |
| Fan or BEV tracing space | Ray-tracing volume used by RTD kernels | `fanCorner_mm=(cpbCorner.x*lenToMm, cpbCorner.y*lenToMm, startZ_mm)`, `fanDelta_mm=(cpbRes.x*lenToMm, cpbRes.y*lenToMm, -abs(stepLength_mm))` | mm | `DensityAndSpTracerParams`, `fillBevFromWeqVolumeKernel`, `fillIddAndSigma` | Imported WEQ can fill this volume directly without HU-to-density or HU-to-SP tracing. |
| Profile texture space | 3D texture indexed as `(channel, depth, profileRow)` | `channel` fastest, then depth, then energy or LUT row | profile axis units from `profileSetting`; row axis is LUT row index | `overrideRSigmaFromCarbonProfileKernel`, profile audits | `profileRowIdx` can come from actual layer index, LUT energy interpolation, or fallback inference depending on runtime shape logic. |
| Subspot texture space | 3D texture indexed as `(channel, subspot, energyRow)` | `channel` fastest, then subspot index, then energy row | channel semantics use beam-plane offsets and sigma units; row axis follows LUT energy rows unless remapped | Legacy CarbonPBS kernels and current wrapper subspot remap path | See the dedicated `subspotData` layout tables below. |

## `subspotData` Texture Layout Table

### Axis and flattening contract

| Element | Legacy CarbonPBS contract | Current RTD compatibility contract | Audit notes |
| --- | --- | --- | --- |
| Tensor shape | `(nEnergyRows, nSubspots, 5)` | Pybind requires `(num_rows, max_subspots_per_layer, 5)` | Current pybind accepts `num_rows != actual_num_layers` and remaps by energy if needed. |
| Width or fastest axis | `5` channels | `5` channels | This is stable across legacy CarbonPBS, `src/tests/carbonpbs_data_layout_example.cu`, and the current wrapper struct. |
| Height axis | `isubspot` | `subspot index` | Stable. |
| Depth axis | `eneIdx` from `enelist` and `binarySearchEneIdx(...)` | Either actual layer row when counts match, or remapped fractional LUT row projected onto actual `layer_energy` | This is where the current compatibility path diverges most from legacy CarbonPBS. |
| Flattening rule | `base = (energyRow * nSubspot + subspotIdx) * 5 + channel` | Same flattened storage in `RTDBeamSettings.subspotData` | Verified by `src/tests/carbonpbs_data_layout_example.cu` and `include/core/raytracedicom_integration.h`. |
| Texture creation | Point-sampled 3D texture, clamp addressing, unnormalized coordinates | Legacy CarbonPBS uses texture objects; current wrapper keeps flattened host storage and, in legacy CPB fallback mode, creates the equivalent 3D texture | The exact texture path is still available in the legacy CPB fallback branch of the wrapper. |

### Channel semantics

| Channel index | Semantic | Legacy read site | Current use | Audit notes |
| --- | --- | --- | --- | --- |
| `0` | `deltaX` | `tex3D(subspotData, 0.f, isubspot, eneIdx)` | Remapped into `beam.subspotData[base + 0]` and later added to `rawSpotLattice.spotOffset.x` during exact RTD convolution | Offset is divided by `lenToMm` before use in the current wrapper. |
| `1` | `deltaY` | `tex3D(subspotData, 1.f, isubspot, eneIdx)` | Remapped into `beam.subspotData[base + 1]` and later added to `rawSpotLattice.spotOffset.y` | Same unit handling as `deltaX`. |
| `2` | `weight` | `tex3D(subspotData, 2.f, isubspot, eneIdx)` | Controls whether the current wrapper accumulates a subspot at all and scales the copied dense spot plane before convolution | Zero-weight subspots are skipped in both paths. |
| `3` | `sigmaX` | `tex3D(subspotData, 3.f, isubspot, eneIdx)` | Used to derive per-subspot entry sigma for exact RTD convolution | Current wrapper applies a cm-like heuristic when `lenToMm > 1` and sigma is very small. |
| `4` | `sigmaY` | `tex3D(subspotData, 4.f, isubspot, eneIdx)` | Used exactly like `sigmaX` for the y direction | Stable channel meaning. |

Current fixture-specific notes:

- `subspot_data.shape = (123, 1, 5)`, so the exported LUT has one subspot per LUT energy row.
- `layer_energy.shape = (29,)`, so the current pybind path remaps `subspot_data` from 123 LUT rows onto 29 actual plan layers.
- The current wrapper therefore never consumes the fixture `subspot_data` rows as a direct `layer_energy`-aligned tensor.

## `dosecal.py` and `test_data` Field-To-Consumer Mapping

### Direct `dosecal.py -> cuFinalDose(...)` compatibility arguments

The call sequence in `tps_py/dosecal.py` is the direct baseline for the compatibility path used by the fixture loader in `test_data/rtd_test_case_bg800_beam925.py`.

| `dosecal.py` field | Fixture key or export | Legacy CarbonPBS consumer | Current compatibility consumer | Audit notes |
| --- | --- | --- | --- | --- |
| `finalDose` | `finalDose` | Output dose array for `cudaFinalDose(...)` | `run_carbonpbs_final_dose(...)` validates size and writes RTD dose into the provided array | Current `cuCalDose3` compatibility wrapper does not populate CSC outputs from this dose. |
| `rayweq` | `rayweq`, `water_equivalence` | WEQ header and payload textures | `build_carbonpbs_context(...)`, wrapper WEQ import path | Live and authoritative in both paths. |
| `roiIdx` | `roiIdx` | ROI xyz triplets | `to_roi_linear_indices(...)` | Live in both paths, but current wrapper converts to linear indices immediately. |
| `all_energies` | `all_energies` | Per-spot energy input | Length-check only in pybind | Current wrapper effectively ignores per-spot energies beyond cardinality validation. |
| `sourcePos` | `sourcePos`, `source_pos` | Reads first source position and sets source geometry | Reads first source position only | Per-spot variation is collapsed. |
| `tmpBeamDir` | `tmpBeamDir`, `beam_dir` | Per-spot beam directions | Stored as `spotBeamDirections`, mean direction inferred | Live in both paths. |
| `bmxdir` | `bmxdir`, `beam_xdir` | Beam x-axis basis | Wrapper basis construction | Live in both paths. |
| `bmydir` | `bmydir`, `beam_ydir` | Beam y-axis basis | Wrapper basis construction | Live in both paths. |
| `doseGrid.corner` | `doseGrid_corner` | Grid geometry | Dummy CT and dose geometry | Live in both paths. |
| `doseGrid.resolution` | `doseGrid_resolution` | Grid geometry | Dummy CT and dose geometry; `lenToMm` heuristic | Live in both paths. |
| `doseGrid.dims` | `doseGrid_dims` | Grid geometry | Dummy CT allocation and output validation | Live in both paths. |
| `longitudalCutoff` | `longitudalCutoff`, `longitudal_cutoff` | Direct WEQ cutoff in device kernels | Collapsed to one value per layer, then mapped through RTD cutoff logic | Semantic drift is explicit here. |
| `enelist` | `enelist`, `energy_list` | LUT row selector | LUT row selector plus profile/subspot remap axis | Live in both paths. |
| `idddata` | `idddata`, `idd_data` | IDD texture | Converted to CIDD and used as RTD energy table | Live in both paths, but current path rebuilds the table. |
| `iddsetting` | `iddsetting`, `idd_setting` | Depth axis for `idddata` | Depth axis and scale-fact builder for RTD energy data | Live in both paths, but current path applies extra unit heuristics. |
| `profiledata` | `profiledata`, `profile_data` | Profile texture | Optional Carbon profile override texture in wrapper | In the current path this entire branch is optional and shape-inferred. |
| `profilesetting` | `profilesetting`, `profile_setting` | Profile depth axis | Profile texture shape and depth axis | Live in both paths. |
| `beamparadata` | `beamparadata`, `beam_para_data` | `(r2, rtheta, theta2)` per LUT row | Same coefficients, but interpolated by `profileRowIdx` when override is enabled | Live in both paths. |
| `subspotdata` | `subspotdata`, `subspot_data` | Subspot texture | Exact RTD convolution inputs or legacy CPB fallback texture | Live in both paths, but current path may remap rows. |
| `layerInfo` | `layerInfo`, `layer_info` | Layer loop sizes | Layer loop sizes and cutoff reduction | Live in both paths. |
| `layerEnergy` | `layerEnergy`, `layer_energy` | `binarySearchEneIdx(...)` lookup target | Actual wrapper energy list and energy interpolation target | Live in both paths. |
| `idbeamxy` | `idbeamxy` | Spot position on WEQ lattice | Spot positions, raw spot lattice, decoded spot bounds, source-distance inference | This is one of the most fragile contracts in the current path. |
| `npermu * beam['weight_vector'] / nFrac` | `nPar`, `number_particle` | Spot weights in `cudaFinalDose(...)` | Stored as `spotWeights` for current `calcDose`, `cuFinalDose`, and current `cuCalDose3` compatibility wrapper | Legacy `cudaCalDose3(...)` did not accept this argument. |
| `sad` | `sad` | Source-axis distance | `beamSettings.sad` and `sourceDist` initialization | Live in both paths. |
| `cutoff` | `cutoff` | `transCutoff` in CarbonPBS device kernels | Ignored by current pybind compatibility wrappers | This is a silent compatibility change. |
| `beamParaPos` | `beamParaPos` | Passed into CarbonPBS kernels | `beamSettings.beamParaPos` for optional profile override | Conditionally live in current path. |
| `gpuId` | `gpuId` | `cudaSetDevice(gpuId)` | `subsecondWrapper(..., gpu_id, ...)` | Live in both paths. |

### Additional exported `test_data` fields and their current consumers

| Exported field | Export note in `variable_name_meta.csv` | Actual current consumer | Status in current contract |
| --- | --- | --- | --- |
| `spot_energy_sigmaxy` | Derived from `all_energies` and interpolated `subspot_data[:,0,3/4]` | `src/tests/wrapper_integration_test.cu::loadSpotSigmasPerSpot(...)` | Test-harness only; not consumed by the pybind compatibility path. |
| `spot_sigmax` | Derived from `spot_energy_sigmaxy[:,1]` | `src/tests/wrapper_integration_test.cu::loadSpotSigmasPerSpot(...)` fallback branch | Test-harness only. |
| `spot_sigmay` | Derived from `spot_energy_sigmaxy[:,2]` | `src/tests/wrapper_integration_test.cu::loadSpotSigmasPerSpot(...)` fallback branch | Test-harness only. |
| `final_dose` | Reference output dose | Test-side comparison or offline inspection | Not an input to the runtime compatibility path. |
| `ext_contour_linear` | Body ROI linear indices | `src/tests/wrapper_integration_test.cu::createCtDataFromRoiMask(...)` when building a ROI-derived water phantom | Test-harness only; not part of the current pybind compatibility signature. |
| `ext_contour_linear_opt` | Optimization ROI linear indices | No consumer in the audited runtime path | Exported but unused in the current compatibility path. |
| `ctgrid_data` | CT grid data after `makeCTGrid` | No consumer in the audited compatibility path | Exporter-only context. |
| `dosegrid_data` | Dose grid data used for WEQ tracing | No direct consumer in the audited compatibility path | Indirectly replaced by the exported `water_equivalence` volume. |

### `/tables` consumer mapping

| Table path | Actual current consumer | Used for in the current compatibility path | Not used for in the current compatibility path |
| --- | --- | --- | --- |
| `tables/proton_cumul_ddd_data.txt` | `src/utils/energy_reader.cpp`, called from `build_energy_from_carbonpbs(...)` | Supplies reference `peakDepths` and, in direct RTD wrapper usage, can supply reference CIDD and scale facts | Current CarbonPBS pybind path does not use this table to build `ciddMatrix`; it builds CIDD from exported `idddata` instead. |
| `tables/density_Schneider2000_adj.txt` | `energyReader(...)` | Density lookup vector for BEV tracing or material setup | Not part of the original CarbonPBS exported arrays. |
| `tables/HU_to_SP_H&N_adj.txt` | `energyReader(...)` | Stopping-power lookup vector | Not part of the original CarbonPBS exported arrays. |
| `tables/radiation_length.txt` | `energyReader(...)` | Radiation-length lookup vector in the non-`WATER_CUBE_TEST` build path | Not part of the original CarbonPBS exported arrays. |
| `tables/radiation_length_inc_water.txt` | `energyReader(...)` when `WATER_CUBE_TEST` is defined | Alternative radiation-length LUT | Not otherwise used in the default audited path. |
| `tables/generated/README.md` | Human documentation only | Documents generated LUT provenance and ranges | Not a runtime consumer. |
| `tables/generated/proton_from_rtd_for_carbonpbs_info.json` | Human documentation only | Documents RTD proton LUT energy range (`62.3866` to `226.638`) and peak-depth range | Not a runtime consumer, but highly relevant to audit the current energy-axis mismatch risk. |

## Deviations From `RayTracedicom_main`

The table below records the material deviations in scope relative to `RayTraceDicom-main (1)/RayTraceDicom-main/src/kernel_wrapper.cu`, `gpu_convolution_2d.cu`, and `beam_settings.h`.

| Area | Current repository behavior | Upstream RTD main baseline | Impact |
| --- | --- | --- | --- |
| Legacy CarbonPBS module | `carbonPBS/cudaCalDose.cpp` and `carbonPBS/deviceCalDose.cu` remain present and expose `cuCalDose3`, `cuFinalDose`, `cuCalDoseNorm`, `cuCalFluenceMapAlphaBeta`, `cuFinalDoseAndRBEMap`, `cuCalFluenceMapRBE`, and `cuRotate3DArray` | No equivalent CarbonPBS compatibility module exists upstream | The repository now has a full legacy compatibility contract in addition to RTD. |
| Beam settings data model | `include/core/raytracedicom_integration.h` defines flat `RTDBeamSettings` with CarbonPBS-specific fields: `subspotData`, `layerSpotCounts`, `spotPositions`, `spotBeamDirections`, `waterEquivalence`, `profileData`, `beamParaData`, `layerLongitudinalCutoffs`, and ROI indices | Upstream `beam_settings.h` expects dense `HostPinnedImage3D<float>* spotWeights`, beam energies, sigmas, spacing, source distances, and caller-supplied transforms | The current wrapper reconstructs RTD inputs from CarbonPBS arrays rather than receiving the upstream RTD data model directly. |
| Python API surface | `src/bindings/raytracedicom_pybind.cpp` exposes `calcDose`, `cuFinalDose`, and `cuCalDose3` with CarbonPBS-style argument lists | Upstream RTD main does not ship a CarbonPBS-style pybind compatibility layer | Binding-side contract drift is now as important as wrapper-side drift. |
| `cuCalDose3` behavior | Current pybind `cuCalDose3` zeros CSC outputs, computes final dose through RTD, and returns a dict containing `dose_grid` plus zeroed CSC arrays | Legacy CarbonPBS `cuCalDose3` computes a CSC sparse spot-dose matrix; upstream RTD has no such compatibility shim | `cuCalDose3` is no longer behaviorally equivalent to the original CarbonPBS API. |
| `cuCalDose3` signature | Current pybind `cuCalDose3` requires `num_particles_per_beam` | Legacy `carbonPBS/cudaCalDose.cpp::cudaCalDose3(...)` does not accept that argument | This is a live call-signature break relative to the legacy CarbonPBS contract. |
| `cutoff` handling | Current pybind compatibility wrappers accept `cutoff` for API compatibility but ignore it | Legacy CarbonPBS device kernels use `transCutoff` to reject very small lateral Gaussian weights | Lateral support can differ even when all other inputs are identical. |
| CT handling in compatibility path | Current pybind compatibility wrappers allocate a dummy CT filled with `1000.0f` and rely on imported WEQ | Upstream RTD expects actual CT data and derives BEV density and cumulative stopping power by tracing through the image | The current compatibility path is not a thin wrapper around upstream RTD; it is a different geometry/material pipeline. |
| Direct WEQ import | Wrapper can import full `waterEquivalence` and fill BEV density and cumulative stopping power via `fillBevFromWeqVolumeKernel` | Upstream `kernel_wrapper.cu` uses `fillBevDensityAndSp(...)` on CT and lookup textures | Any WEQ header or layout mismatch now moves dose directly, without CT tracing to mediate it. |
| Spot-position semantics | Wrapper interprets `idbeamxy` as WEQ-lattice indices and reconstructs a dense spot grid with `+0.5`, `floor(raw)`, and decoded bounds logic | Upstream RTD expects dense spot weight maps and an explicit `spotIdxToGantry` transform from the caller | This is one of the largest geometry deviations in the current codebase. |
| Beam-basis construction | Wrapper derives `bmZ`, `gX`, `gY`, `gZ`, source position, and isocenter from exported beam basis vectors and mean beam direction | Upstream RTD consumes caller-supplied affine transforms and does not reconstruct the beam basis this way | Coordinate correctness now depends on extra wrapper logic that has no upstream counterpart. |
| Unit heuristics and environment overrides | Wrapper uses `lenToMm`, `RTD_ENERGY_DEPTH_UNIT`, `RTD_ENERGY_DEPTH_SCALE`, sigma heuristics, and other runtime checks | Upstream RTD assumes the caller has already supplied self-consistent units | The current repository has a heuristic unit-repair layer that can change execution semantics. |
| Longitudinal cutoff semantics | Current pybind collapses per-spot `longitudal_cutoff` to one value per layer; wrapper later maps that to `peakDepth = cutoff / BP_DEPTH_CUTOFF` | Legacy CarbonPBS compares `weqDepth < longitudalCutoff` directly; upstream RTD has no CarbonPBS `longitudal_cutoff` concept | Depth stopping behavior is no longer numerically identical to either upstream RTD or legacy CarbonPBS. |
| Subspot energy-row remap | Current pybind remaps `subspot_data` from LUT rows to actual layer energies when row counts differ | Upstream RTD has no CarbonPBS `subspotData` LUT and no remap logic | The current path tolerates and transforms a mismatch that legacy CarbonPBS resolves by direct LUT indexing. |
| Carbon profile override | Wrapper can infer a profile texture from `profileData` and `beamParaData` and then override `bevRSigmaEff` with CarbonPBS profile parameters | Upstream RTD main has no CarbonPBS profile override path | Lateral transport now depends on an additional model branch that has no upstream reference behavior. |
| Convolution file scope | `src/algorithms/convolution.cu` contains both RTD-style `xConvGathResampGpu` or `yConvGathResampGpu` and added CarbonPBS-specific helpers: subspot extraction, subspot-range calculation, subspot-to-CPB convolution, and CPB-to-ray mapping | Upstream `gpu_convolution_2d.cu` contains only the RTD convolution kernels and `gpuConvolution2D(...)` wrapper | The convolution module is now a hybrid compatibility layer, not a pure upstream RTD file. |
| Exact RTD convolution path | Current wrapper uses `performExactRTDConvolution2D(...)` on a reconstructed dense raw spot lattice when possible | Upstream RTD convolves caller-supplied dense spot weight maps | The current repo preserves the RTD kernels but changes how the dense spot grid is obtained. |
| Energy-table construction | `build_energy_from_carbonpbs(...)` builds CIDD from exported `idddata` but fills density/SP/RRL and reference `peakDepths` from `tables/` | Upstream RTD expects the caller to provide a fully formed energy struct | The compatibility adapter now splices together exported CarbonPBS data and RTD reference tables. |
| Energy-axis clamp risk | `interpolate_peak_depth_from_reference(...)` clamps energies outside the reference table range to the nearest RTD proton peak depth | Upstream RTD main does not need to synthesize CarbonPBS peak depths this way | For fixture energies above the RTD proton range, the current path can silently clamp `peakDepths`. |
| Debug and audit surface | Wrapper and pybind add environment-driven diagnostics such as `RTD_INPUT_AUDIT` and `RTD_SIGMA_DEBUG` and print shape, unit, and geometry audits | Upstream RTD main does not include this CarbonPBS-specific debug surface | These diagnostics are useful, but they also reflect the existence of many non-upstream execution branches. |

## Ranked Likely Causes Of Bragg Peak Position Error

This ranking is an inference from the current source and fixture contracts, not a runtime proof. The ordering prioritizes the mechanisms that most directly change the WEPL-to-depth mapping used by the current compatibility path.

| Rank | Likely cause | Why it can move the Bragg peak | Source evidence |
| --- | --- | --- | --- |
| `1` | `longitudal_cutoff` is no longer applied with legacy CarbonPBS semantics | Legacy CarbonPBS stops contribution when `weqDepth < longitudalCutoff`. The current path collapses per-spot cutoffs to one layer value and then inverts RTD logic with `peakDepth = cutoff / BP_DEPTH_CUTOFF`, which changes where transport is terminated. | `src/bindings/raytracedicom_pybind.cpp::derive_layer_longitudinal_cutoffs(...)`, `src/core/raytracedicom_wrapper.cu` around `layerLongitudinalCutoffMm(...)`, `peakDepth = layerCutoffMm / BP_DEPTH_CUTOFF`, and `afterLastStep` handling. |
| `2` | WEPL-to-IDD scaling now depends on pybind or wrapper unit heuristics instead of one explicit exporter contract | The current path computes `energyScaleFact` from `iddsetting`, then applies `iddDepthUnitToMm` and later `energyDepthToMm`. Any wrong mm or cm assumption shifts the depth index into the CIDD texture and therefore shifts the Bragg peak position directly. | `src/bindings/raytracedicom_pybind.cpp::build_energy_from_carbonpbs(...)` and `src/core/raytracedicom_wrapper.cu` around `energyScaleFact = energyScaleFact_table / energyDepthToMm`. |
| `3` | The current wrapper can bypass upstream CT and SP tracing entirely and trust the imported WEQ volume | If the WEQ payload layout, header spacing, or X or Y alignment is even slightly wrong, the current path accumulates dose against the wrong cumulative stopping-power axis before any RTD IDD lookup happens. | `src/core/raytracedicom_wrapper.cu::fillBevFromWeqVolumeKernel(...)`, `hasWeqVolume`, and the direct BEV fill branch. |
| `4` | `idbeamxy` semantics are internally inconsistent in the current wrapper | The fixture states that `idbeamxy` is index-like, but the wrapper uses `raw`, `raw-0.5`, and `floor(raw)` variants in different places. That can move which WEQ rays are sampled and which spot lattice cells are populated, especially in heterogenous WEQ fields. | `decodeSpotPosition(...)`, `buildRawSpotLattice(...)`, and `estimateVirtualSourceDistancesFromSpots(...)` in `src/core/raytracedicom_wrapper.cu`. |
| `5` | The current compatibility path mixes three energy axes: actual plan layers (`29`), LUT rows (`123`), and profile or beam-parameter rows (`123`) | The wrapper interpolates actual layer energies against `enelist` for CIDD lookup, profile-row selection, and subspot remap, but not all tables are remapped the same way. Any mismatch between these axes can distort the layer-wise depth model. | `build_carbonpbs_context(...)`, `remap_subspot_lut_to_actual_layers(...)`, and wrapper `profileRowIdx` selection logic. |
| `6` | The current pybind path synthesizes `peakDepths` from RTD reference proton tables whose energy range stops at `226.638`, while the fixture layer energies rise to `359.44` | When energies exceed the reference LUT range, `interpolate_peak_depth_from_reference(...)` clamps to the end of the proton table. Even when `longitudal_cutoff` overrides peak depth later, this still affects fallback behavior, sigma-air coefficients, and any path that relies on synthesized `peakDepths`. | `tables/generated/proton_from_rtd_for_carbonpbs_info.json`, fixture energy ranges, and `src/bindings/raytracedicom_pybind.cpp::interpolate_peak_depth_from_reference(...)`. |
| `7` | Carbon profile override and beam-parameter interpolation change RTD lateral transport relative to upstream | This is less direct than a depth-axis mismatch, but any change to `bevRSigmaEff` and entry sigma changes how energy is distributed through the fan and can bias the apparent peak position after superposition and dose transfer. | `src/core/raytracedicom_wrapper.cu::overrideRSigmaFromCarbonProfileKernel(...)` and the `profileTex` or `beamParaData` branch. |

## Risks / Trade-offs

- [Risk] This is a static audit and not a runtime reproduction. -> Mitigation: every ranked cause is tied to concrete source sites so implementation follow-up can target the most direct contract changes first.
- [Risk] Some fields, especially `iddsetting` units and `longitudal_cutoff` invariants, may vary by exporter configuration. -> Mitigation: the dossier records the current fixture evidence and identifies where the code currently infers or collapses those semantics.
- [Risk] The repository now contains both legacy CarbonPBS and current RTD compatibility paths, which can tempt future work to mix intended semantics with current behavior. -> Mitigation: this dossier keeps the contracts separated explicitly and names where they diverge.

## Migration Plan

1. Accept this audit-only change as the baseline description of the current repository behavior.
2. Use the variable, coordinate, texture, consumer, and deviation tables to define a follow-up implementation change.
3. In the follow-up change, fix one contract mismatch class at a time, starting with the top-ranked Bragg peak depth causes.
4. Keep this change documentation-only; any numerical or API modification belongs in later work.

Rollback strategy:

- If the project decides not to use this audit as the baseline, revert only the OpenSpec artifacts. No runtime code changes are introduced here.

## Open Questions

- Does the exporter guarantee that `longitudal_cutoff` is constant within a layer, or is the current layer-first collapse already lossy for valid plans?
- Are `iddsetting` and `profilesetting` guaranteed to be in mm for all CarbonPBS exports, or are the current wrapper heuristics expected to support mixed unit conventions?
- Is `all_energies` intentionally redundant with `layer_energy`, or is the current pybind path dropping information that later fixes will need to preserve?
