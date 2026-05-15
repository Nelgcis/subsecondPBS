# Generated LUTs (from RayTraceDicom reference tables)

This folder contains LUTs converted from the reference RayTraceDicom text table:

- `../proton_cumul_ddd_data.txt`

## Files

- `proton_from_rtd_for_carbonpbs.npz`
  - A **CarbonPBS-friendly** packed numpy archive containing:
    - `enelist` (same as `energiesPerU`)  shape `(nEnergies,)`
    - `cidd_rtd` (raw RTD cumulative table) shape `(nEnergies, nSamples)`
    - `cidd_resampled_mm` (resampled CIDD on a *global* depth axis) shape `(nEnergies, nSamples)`
    - `idd_segment_resampled` (segment-integral IDD = diff(CIDD)) shape `(nEnergies, nSamples)`
    - `iddsetting_cm = [start_cm, step_cm, nSamples]`

- `proton_from_rtd_for_carbonpbs_info.json`
  - Quick summary: `nSamples`, `nEnergies`, min/max energy, depth axis settings.

## How to regenerate

From repo root:

```bash
python3 tools/convert_rtd_cidd_to_carbonpbs_npz.py \
  --input tables/proton_cumul_ddd_data.txt \
  --output tables/generated/proton_from_rtd_for_carbonpbs.npz \
  --info_json tables/generated/proton_from_rtd_for_carbonpbs_info.json
```

## Notes on dimensions

For the reference table `proton_cumul_ddd_data.txt`:

- `nSamples = 1024` (depth samples)
- `nEnergies = 147` (energy axis samples)

`energiesPerU` is the **machine energy axis** for interpolation.
Plan energy layers (e.g. 32 layers) are **not** required to match `nEnergies`.
