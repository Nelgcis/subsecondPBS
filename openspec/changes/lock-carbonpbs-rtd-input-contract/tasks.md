## 1. Contract Baseline

- [ ] 1.1 Consolidate the current CarbonPBS, `test_data`, and RTD-wrapper variable crosswalk into one source-of-truth contract document.
- [ ] 1.2 Mark every field as `consistent`, `ambiguous`, `shape-risk`, `unit-risk`, `order-risk`, `rename-needed`, or `unknown`, with source-backed justification.

## 2. Canonical Input Model

- [ ] 2.1 Define the canonical internal ROI representation and the boundary normalization rules for accepted Python ROI input shapes.
- [ ] 2.2 Define the canonical `subspotData` contract, including channel meanings, row-axis metadata, remap rules, and RTD exact-convolution mapping.
- [ ] 2.3 Define the canonical beam-geometry contract for `beamDirection`, `bmxdir`, `bmydir`, `source`, `sad`, and reference-plane semantics.
- [ ] 2.4 Define the canonical `doseGrid` and `ctGrid` metadata contract, including `(x,y,z)` metadata order and explicit ndarray-order requirements.
- [ ] 2.5 Define the canonical `rayweq` file contract: the 9-value header declares the authoritative used payload length `nx*ny*nStep`, full versus body-only exports must be explicit, and any reserved-capacity zero tail beyond the header-declared used prefix must either be rejected or normalized instead of being silently treated as the true file shape.

## 3. Parity And Risk Audit

- [ ] 3.1 Compare `dosecal.py` or `test_data` exports with the actual legacy CarbonPBS consumers and the actual RTD-wrapper readers for every major field.
- [ ] 3.2 Produce the high-risk mismatch list and the checklist of fields that already match, fields that need rename or shape work, and fields that remain `unknown`.
- [ ] 3.3 Audit the checked-in `water_equivalence` fixtures and metadata for prefix parity, zero-tail padding, and byte-length consistency, and record that the current files preserve the same header-declared valid prefix but disagree on trailing zero padding and on the metadata-reported `float32` shape.

## 4. `calDoseSubsecond` Integration Gate

- [ ] 4.1 Define the normalized future `calDoseSubsecond` input contract and the rule that ambiguous or unproven inputs must not be silently inferred.
- [ ] 4.2 Implement boundary validation and adapter updates in follow-up work without changing numerical algorithm behavior.
