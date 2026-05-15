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

## 3. Numerical Error Accumulation Risk

- [ ] 3.1 Audit possible error accumulation paths in subSecond: WEQ sampling, CIDD differencing, density/radiation-length lookup, sigmaSq evolution, mass/stepVol scaling, superposition truncation, and atomic accumulation.
- [ ] 3.2 Define diagnostics that distinguish physical dose differences from accumulated numerical/contract error.

## 4. CUDA 12.1 Optimization Exploration

- [x] 4.1 Inventory CUDA 12.1-relevant options for subSecond after correctness is stable: memory pools, async allocation, graph capture, improved stream scheduling, texture/data reuse, occupancy tuning, and kernel fusion opportunities.
- [x] 4.2 Prioritize optimizations by expected impact and risk to dose correctness.
