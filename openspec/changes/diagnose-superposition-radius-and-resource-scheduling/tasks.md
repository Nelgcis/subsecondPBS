## 1. Superposition Radius Diagnosis

- [x] 1.1 Map the full `rSigmaEff -> tileRadCalc -> overflow` chain and document each input variable that can reduce `rSigmaEff`.
- [x] 1.2 Compare CarbonPBS final-dose lateral-spread inputs (`profileData`, `beamParaData`, `subspotData`, `rayweq`) with subSecond RTD lateral-spread inputs (`entrySigmaSq`, `sigmaSq`, `voxelWidth`, `density`, `rRadiationLength`, `raySpacing`).
- [x] 1.3 Define a per-layer radius attribution report that can run on existing plans without generating new plans.
- [x] 1.3a Add a superposition overflow debug context to the primary warning path so every `Primary superposition required radius larger than max radius` line can include layer index, energy, energy-table index, peak depth, longitudinal cutoff, active step range, ray/superposition dimensions, and representative voxel-width/sigma limit values.
- [x] 1.4 Add or identify diagnostics for CT/WEQ factors: HU-density/SP mapping, WEQ header start/step, active WEQ depth span, density/radiation-length summaries, voxel/fan spacing, and coordinate/unit consistency.
- [x] 1.5 Add or identify diagnostics for beam/layer factors: energy, peak depth, longitudinal cutoff, spot sigma, source distance, ray spacing, CPB dimensions, spot counts, and layer active-step ranges.
- [x] 1.5a Add residual positive-IDD overflow sample diagnostics for cumulative SP, density, mass, range-stop relation, positive-IDD ranges, and heuristic cause hints.
- [ ] 1.6 For at least one overflowing plan and one non-overflowing plan, produce a layer-sorted table ranking likely overflow drivers.
- [x] 1.7 Decide whether overflow is expected physics for specific layers, a geometry/unit contract bug, or a scheduling/algorithm limit that requires an explicit policy.

## 2. Resource Scheduling And CarbonPBS/SubSecond Runtime Comparison

- [x] 2.1 Explain why current subSecond runtime is lower or higher than CarbonPBS by stage: input preparation, ray tracing, IDD/sigma, superposition, transfer, allocation, texture setup, and synchronization.
- [x] 2.2 Identify uncleaned performance debt outside superposition: repeated allocations, texture recreation, host-device copies, stream synchronization, debug/audit copies, padding, and fallback branches.
- [ ] 2.3 Summarize CarbonPBS advantages and subSecond advantages under the same plan/input contract.
- [x] 2.4 Diff subSecond against upstream `RayTraceDicom-main` for capabilities subSecond regressed on or avoided: BEV-dose buffer lifetime, batched superposition stream-sync, convolution alloc/free pattern, multi-stream plumbing, texture-pool wiring, `__launch_bounds__`, build-flag drift between Makefile and CMake. See design.md section "Upstream RayTraceDicom-main vs subSecond Capability Diff".

## 3. Numerical Error Accumulation Risk

- [x] 3.1 Audit possible error accumulation paths in subSecond: WEQ sampling, CIDD differencing, density/radiation-length lookup, sigmaSq evolution, mass/stepVol scaling, superposition truncation, atomic accumulation, fast-math intrinsic ULPs, texture bilinear precision, axis-aligned voxel-boundary effects, and magic constants (`sigmaDelta=0.21`, `RAY_WEIGHT_CUTOFF=1e-6`). See design.md section "Numerical Error Accumulation Audit".
- [ ] 3.2 Define diagnostics that distinguish physical dose differences from accumulated numerical/contract error.

## 4. CUDA 12.1 Optimization Exploration

- [x] 4.1 Inventory CUDA 12.1-relevant options for subSecond after correctness is stable: memory pools, async allocation, graph capture, improved stream scheduling, texture/data reuse, occupancy tuning, and kernel fusion opportunities. Detailed inventory O1-O15 in design.md section "CUDA 12.1 Optimization Inventory For cuFinalDose".
- [x] 4.2 Prioritize optimizations by expected impact and risk to dose correctness. Recommended order: O1 (per-beam BEV dose buffer) > O12 (Makefile `--use_fast_math` parity) > O5 (texture pool wiring) > O2 (`cudaMallocAsync`) > O3 (CUDA Graph capture) > O8 (`__launch_bounds__`) > O4 (multi-stream primary vs nuclear) > O11 (L2 residency hints) > O7 (cp.async) > O6 (warp shfl reduction) > O14 (cooperative groups). Reject for sm_86: O9 (PDL), O10 (TMA), O13 (fp16 density), O15 (megakernel).
- [x] 4.3 Lock target compute capability and tag each optimization with the sm_* gate: sm_86 (Ampere, current build) excludes TMA/DSMEM/PDL (sm_90); these belong to a separate change if hardware migration is ever in scope.

## 5. Spatial Filter Architecture (BEV-Fan Box vs ROI-List)

- [x] 5.1 Map the spatial filtering used by each engine: subSecond uses a BEV-fan AABB projected back to dose-grid index space with a ±MAX_SUPERP_RADIUS ring; carbonPBS iterates `roiIndex[]` and culls per-spot by Gaussian threshold and depth cutoff. `roiIdx` is the sole spatial gate for carbonPBS but is ignored by subSecond in production (`calculateROIRangeKernel` is dead code). See design.md section "Beam-Fan Box vs ROI-List Spatial Filter".
- [x] 5.2 Document cost-curve orthogonality: subSecond cost is `O(rayDims × layers)` regardless of ROI size; carbonPBS cost is `O(num_roi × num_spots × nsubspot)` regardless of beam geometry.
- [ ] 5.3 Decide on optimizations O-DOSE-1..5 (box intersection with ROI bbox, per-beam box hoist, ROI-mask strict gating, fan-vs-3σ extent reconciliation with H8, dead-code cleanup).

## 6. Pybind Dose Deficiency Investigation (Halo Missing, Shallow Zero)

- [x] 6.1 Confirm the nuclear correction toggle is compile-time-gated by `NUCLEAR_CORR` macro AND runtime-gated by the pybind `nuclear_correction` argument; document why no environment variable can enable halo from Python without rebuilding.
- [x] 6.2 Document the side effects of toggling `nuclear_correction` on primary dose (not just halo): `eRefSq`, `sigmaDelta`, `suppressDistalSigmaDip` all change in `src/algorithms/idd_sigma.cu:105-148`.
- [x] 6.3 Diagnose shallow-region dose=0 root causes: `firstMaterial+1` off-by-one at `src/core/raytracedicom_wrapper.cu:2043`, dose-box z-range clipping at `beamFirstInside` (`raytracedicom_wrapper.cu:5527-5564`), and `mass > 1e-2f` cutoff at `src/algorithms/idd_sigma.cu:197`. WEQ header start trap covered separately.
- [ ] 6.4 Verifier-grade dose-diff: with `-DNUCLEAR_CORR=GAUSS_FIT` and `nuclear_correction=True`, compare pybind cuFinalDose vs carbonPBS on a single-beam single-layer water plan; confirm halo contribution and shallow-region recovery.
- [x] 6.5 Document the operating contract for Python callers: the rebuild command, the `dose_cal_config["nuclear_correction"]` flag, and the `rtdSupportMatrix()` self-check. See design.md section "Enabling Nuclear Correction From Python".
- [x] 6.6 Quantify the `/output` dose-support deficit against CarbonPBS for `bg879_beam1035`: RTD/TPS sum ratio is `0.671988`, RTD bbox is `[88,15,150]..[144,93,206]`, TPS bbox is `[78,1,140]..[154,146,216]`, and y=1..14 are exactly zero in RTD while TPS remains positive. See design.md section "Dose Support Clipping Against CarbonPBS Output".
- [ ] 6.7 Fix primary transfer support clipping without changing physics constants or nuclear logic: decouple transfer launch lower bound from `beamFirstInside`, audit `startIdx/maxIdx`, and verify y=1..14 plus the final-dose bbox recover before touching halo.
- [x] 6.8 Carry explicit PB spacing into a per-layer halo contract grounded only in CarbonPBS and `dosecal.py`: derive `layerSpotDelta` from `spotSpacingX/Z` and `layerInfo`, but do not treat spacing as a complete lattice-origin contract. `idbeamxy` / decoded `x_all,z_all` are the authoritative spot-center positions.
- [ ] 6.9 Verify the two fixes independently: first primary-only support against CarbonPBS bbox/per-y support, then nuclear-on lattice with the `[9.24] halo/primary lattice ratio` warning gone and per-layer deltas matching Python spacing (`4.3 mm` early, `3.1 mm` late).
- [x] 6.10 Remove the incorrect explicit-lattice assumption introduced during 6.8: do not require every spot in a layer to satisfy `coord = minCoord + integer * layerSpotDelta`. The observed failure `frac=(0,0.511628)` only proves that `minCoord` is not a valid origin; it must not be interpreted as evidence for an unconfirmed hex/staggered CarbonPBS path.
- [x] 6.11 Rebuild halo rayweight placement from existing CarbonPBS inputs: use decoded `idbeamxy` spot centers and `nPar` weights as the placement source, use `spotSpacingX/Z` only for physical PB spacing / `spotDistInRays` / nuclear normalization, and avoid any point-cloud or WEQ-header spacing inference when explicit spacing exists.
- [x] 6.12 Add audits to answer whether the rayweight-defined halo actually contributes: for representative layers print `sum_rayWeight`, `sum_nucRayWeight`, `spotDistInRays`, `nucWeight` sample range, `devNucIdd/devNucRSigmaEff` nonzero ranges, `SUPERP_OUTPUT_NUC_BEV` sum/max, and whether `nucTransfDiv` launches a non-empty dose box.
- [x] 6.13 Compare CarbonPBS source behavior before changing halo placement further: inspect `carbonPBS/deviceCalDose.cu` and related launch code to confirm it consumes `idbeamxy` as continuous spot coordinates and does not implement a separate hex/stagger lattice branch. Any new RTD assumption must cite a matching CarbonPBS or `dosecal.py` data contract.
