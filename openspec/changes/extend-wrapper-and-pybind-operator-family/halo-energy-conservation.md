# Halo Energy-Conservation Diagnosis And Staged Repair Plan

Status: explore-mode artifact; no code changes yet.
Captured 2026-05-08 from openspec-explore session.
Anchors: `src/core/raytracedicom_wrapper.cu`, `src/algorithms/idd_sigma.cu`,
upstream `RayTraceDicom-main (1)/RayTraceDicom-main/src/kernel_wrapper.cu`.

## 1. The observable

With `nuclear_correction=true`, total dose drops below the `nuclear_correction=false`
baseline by approximately `nucWeight × primary_total` (typically 5-15%).
The expected behavior is total dose ~unchanged, with redistribution of energy from a
narrow primary core into a broader nuclear umbrella.

## 2. Where energy actually disappears (corrected from initial framing)

Initial framing (RC1, "sparsity"): primary is reduced over all 4800 CPB rays but
halo is only added at 25 PB cells. This diagnoses *where* the values land in
buffers but does NOT explain the integrated deficit, because the CPB convolution
that creates `rayWeight` from `PB_weight` conserves total weight:

```
Σ_rays rayWeight  ≈  Σ_PB PB_weight       (Gaussian rasterization)
```

So at the IDD step, primary subtracts `nucWeight × Σ_rays rayWeight × dE/mass` and
halo adds `nucWeight × Σ_PB PB_weight × dE/mass / spotDist²`. With the local
`spotDist=1`, those totals are equal *at the IDD ray plane*. Sparsity alone does
not violate energy conservation in this code.

Refined framing (RC2 dominant): The integrated deficit comes from the BEV-to-dose
transfer, not the IDD step. At `raytracedicom_wrapper.cu:5132-5141`:

```cpp
const Float3IdxTransform nucIdxToFan(
    make_vec3f(layerHaloPlan->spotDelta.x * lenToMm,    // <- mm * 10
               layerHaloPlan->spotDelta.y * lenToMm,
               fanDelta_mm.z),
    make_vec3f(layerHaloPlan->spotOffset.x * lenToMm,   // <- mm * 10
               layerHaloPlan->spotOffset.y * lenToMm,
               fanCorner_mm.z));
```

`layerHaloPlan->spotDelta/spotOffset` is sourced from `weqHeader[7]/[4]/[6]/[3]`
which are already in mm (verified in `buildPhysicalPBLatticeView` lines 1535-1540).
Multiplying by `lenToMm=10` again places the halo BEV origin at ~10× the correct
fan-frame position. `nucTransfDiv` then samples `bevNucDoseTex` at coordinates
that fall outside the texture extent on every voxel, hits the
`continue` / border-zero branch, and contributes 0 dose.

Net result: primary dose × `(1 - nucWeight)` + halo × 0 ≈ deficit of
`nucWeight × primary_total`, exactly matching the observed 5-15% drop.

Comparison to upstream `kernel_wrapper.cu:1240`:

```cpp
Float3FromFanTransform nucRayIdxToDoseIdx(
    beam.getSpotIdxToGantry(),       // delta/offset in mm, single transform
    beam.getSourceDist(),
    beam.getGantryToDoseIdx());
```

Upstream applies no extra unit conversion. The transform's delta and offset are
already in mm; halo BEV is placed correctly.

## 3. What RC1, RC3, RC4 actually break

After RC2 is fixed, the integrated halo dose returns to ~correct. RC1/RC3/RC4
then govern the LATERAL PROFILE of the halo, not its total magnitude.

- RC3 (`buildPhysicalPBLatticeView` uses WEQ extents): makes `nucRayDims` match
  the CPB grid (e.g. 80×60) instead of the physical PB grid (e.g. 5×5 padded).
- RC1 (sparse `nucRayWeight`): is a downstream symptom of RC3 — once `nucRayDims`
  is the wrong cardinality, only 25 of 4800 cells carry weight.
- RC4 (`spotDist=1` instead of physical_PB_spacing/CPB_spacing): makes
  `nucRSigmaEff = HALF * 1 * voxelWidth / (...sqrt(σ²+nucσ²)...)` — i.e. the
  halo uses primary's σ formula. The intended upstream behavior multiplies by
  `spotDist`, so on the coarse PB-grid the effective σ in cells is
  `phys_σ / PB_spacing` (correctly tighter on the coarse grid).

Visualization of the post-RC2 state (still wrong on profile):

```
    UPSTREAM nucIdd plane              LOCAL nucIdd plane
    (5×5 PB grid, dense)               (80×60 CPB grid, 25 nonzero)
    ┌─────────┐                        ┌──────────────────┐
    │ ●●●●●   │                        │ ··●·····●·····●·│
    │ ●●●●●   │  σ-stretched           │ ··················│
    │ ●●●●●   │  Gaussian on           │ ··●·····●·····●·│  Gaussian uses
    │ ●●●●●   │  PB-spaced grid        │ ··················│  primary σ
    │ ●●●●●   │                        │ ··●·····●·····●·│
    └─────────┘                        └──────────────────┘
    halo σ in PB-cells = phys_σ/PB_sp  halo σ in CPB-cells = phys_σ/CPB_sp
                                       ⇒ halo lateral spread MATCHES primary
                                         instead of being broader
```

Total integrated dose (after RC2 fix) is the same in both. The user-visible
difference is that local halo "looks like a slight primary attenuation" rather
than the wide nuclear umbrella that should appear in the wings.

## 4. Error accumulation across layers

The wrapper iterates layers serially, each adding to the shared `devDoseVol`.
Per-layer error accumulates additively in the dose grid:

| State | Per-layer δ (signed) | After N layers |
|---|---|---|
| Current (all four broken) | `−nucWeight × primary_layer` | total ≈ `(1−nucWeight) × primary_total`, 5-15% deficit |
| RC2 only fixed | `~0` integrated, lateral profile = primary profile | total dose ~correct, halo broad-tail missing |
| RC2+RC3+RC4 fixed | `~0` integrated AND profile correct | upstream parity |
| RC1/RC3/RC4 fixed without RC2 | `−nucWeight × primary_layer` (unchanged) | 5-15% deficit unchanged |

The accumulation is monotonic in the current state because `(1 − nucWeight)`
is applied identically at every layer. A diagnostic that integrates dose over
the full volume per layer will see the deficit growing linearly with completed
layers; a diagnostic restricted to a single layer at a time will see the same
fractional deficit at every layer. Either is sufficient to confirm RC2.

The risk pattern after RC2 lands but before RC3/RC4: integrated dose looks
right, but the spatial profile in the lateral wings is wrong. A reviewer might
read this as "halo working but slightly miscalibrated" and miss that the
lattice is fundamentally misidentified. Therefore the audit gate must include
a lateral-profile check, not just an integrated check.

## 5. Staged repair plan

```
   STAGE 1 (one-line)         STAGE 2 (medium edit)        STAGE 3 (audit)
   ┌─────────────────┐        ┌──────────────────────┐     ┌──────────────┐
   │ Drop *lenToMm   │        │ Rewrite              │     │ Gate-A/B/C   │
   │ in nucIdxToFan  │ ─────> │ buildPhysicalPB-     │ ──> │ matrix +     │
   │ offset.         │        │ LatticeView around   │     │ lateral      │
   │ (RC2)           │        │ canonical PB         │     │ profile      │
   │                 │        │ transform            │     │ check        │
   │ Audit: integral │        │ (RC3, drives RC1+    │     │              │
   │ E_in/E_out per  │        │  RC4 automatically)  │     │              │
   │ layer (9.49)    │        │ + 9.50 lateral gate  │     │              │
   └─────────────────┘        └──────────────────────┘     └──────────────┘

   Maps to tasks: 9.48        Maps to: 9.39, 9.23,         Maps to: 9.26,
                  9.49        9.31, 9.32, 9.33, 9.34       9.27, 9.35,
                              9.50                          9.36, 9.11c
```

Cannot reorder. RC2 must come first because no other fix changes the integrated
dose. RC3+RC4 must move together because changing `nucRayDims` from CPB-grid
size to padded(5,5) without also fixing the σ formula creates an internally
inconsistent halo branch.

## 6. Audit gates — what catches what

| Audit | RC2 regression | RC3 regression | RC4 regression |
|---|---|---|---|
| Integrated `E_subtracted == E_added` per layer (9.49) | YES (catches) | NO | NO |
| Halo dose ratio vs primary-only baseline | YES | NO (after RC2) | NO (after RC2) |
| `bevNucIdd` occupancy fraction (9.50) | NO | YES | NO |
| Halo lateral FWHM vs upstream prediction (9.50) | NO | YES (cardinality) | YES (σ stretching) |
| `spotDist >= 1.5` fail-fast (9.55) | NO | NO | YES |
| Gate-A/B byte-identity to pre-fix proton baseline (9.51) | YES (regressions to primary) | YES | YES |

Implication: 9.49 alone is necessary but not sufficient. 9.50 plus 9.55 plus
9.51 together cover the four root causes. 9.49 is the cheapest sentry to
prevent RC2 regressing in future cleanups.

## 7. Why upstream design forces this exact stage order

Upstream halo path (`kernel_wrapper.cu:686, 901-911, 941, 1240, 1268`):

1. `nucRayDims = roundTo(spotGridDims, superpTileX/Y)` — cardinality from physical
   PB grid, padded.
2. `nucSpotIdx[primRayDims.x * rayY + rayX] = nucRayDims.x * spotY + spotX` —
   one PB → one nucIdx, fine ray maps to nearest PB via projected gantry coord.
3. `spotDistInRays = beam.getSpotIdxToGantry().getDelta().x / beam.getRaySpacing().x`
   — derived from canonical transforms, in rays.
4. `Float3FromFanTransform nucRayIdxToDoseIdx(beam.getSpotIdxToGantry(), ...)` —
   transform reuses PB-to-gantry mapping directly.

Steps 1-3 source from the same canonical `beam.getSpotIdxToGantry()` transform.
Replicating that locally requires `buildPhysicalPBLatticeView` to derive
`(nx, ny, dx, dy, ox, oy)` from canonical spot positions, NOT from
`weqHeader[8/5/7/4/6/3]`. The current local code's choice to read these from
WEQ header is the proximate cause of RC1, RC3, RC4 simultaneously.

Step 4 uses the same delta/offset already in mm. The local `lenToMm` doubling
breaks the unit contract that upstream relies on. This is RC2.

So the upstream design literally couples RC1/RC3/RC4 into one fix
(`buildPhysicalPBLatticeView` rewrite) and isolates RC2 as a separate concern
(BEV transform unit). Our staging mirrors this coupling.

## 8. Open questions worth confirming before stage-2 implementation

- Are CarbonPBS exports always layer-uniform in the physical PB grid? If yes,
  one beam-global PB transform suffices and 9.21's per-layer phrasing is over-
  defensive. If no, `buildPhysicalPBLatticeView` must accept per-layer cardinality.
- Does any current TPS fixture rely on the broken halo behavior (e.g. tuned
  beam weights to compensate for the 10% deficit)? If yes, the fix will appear
  as a 10% over-dose to that fixture. Audit consumers before flipping.
- The `runtimeNuclearEnabled && layerHaloPlan != nullptr ? layerHaloPlan->spotDelta.x
  : beam.spotDelta.x` pattern at line 3911 silently uses `beam.spotDelta`
  (in cm) when halo is off. After 9.48-9.55 land, audit whether `beam.spotDelta`
  is ever used in the halo-on path through this expression and whether its
  unit (cm vs mm) is consistent with `layerHaloPlan->spotDelta` (mm).
