# Verification data

Per-clip and per-recorder output from `src/validation/phase4_dual_model.py`'s
scoring run against the birdnet v0.1.7 reference (108,069-row CSV), covering
51,720 clips across six recorders.

## Files

- `phase4_results_all_recorders.json` -- aggregate summary across all six
  recorders: agreement percentages, coverage, the FP16-vs-FP32 comparison.
  **Note:** the `models` and `labels` fields originally recorded the absolute
  local filesystem paths used to run the scoring pass. Those have been
  hand-edited to bare filenames (`birdnet_fp32_pruned493.onnx`,
  `birdnet_fp16_pruned493.onnx`, `birdnet_fp32_pruned493.labels.txt`) before
  committing -- provenance (which artifact produced these numbers) is
  preserved, the machine-specific path is not. This is the only edit made to
  this file; every numeric result is untouched.
- `phase4_results_Audio_Moth_{1..6}.json` -- per-recorder summary.
- `phase4_detail_Audio_Moth_6.csv` -- **per-clip** detail for one recorder
  (709 clips): both models' raw and allow-list-restricted top-1 species and
  confidence, and whether each matched the reference.
- `reference_csv_species_allowlist.txt` -- 218 species allowed by the
  reference CSV's own predictions (built by `src/validation/build_allowlist.py`).
- `pruning_keepset_493.csv` -- the 493-species keep-set with per-species
  provenance flags (which of the three source lists included it).

## Two 218-species allow-lists, and why they're not the same file

Two allow-lists of exactly 218 species exist in this repo, from unrelated
derivations:

- `verification_data/reference_csv_species_allowlist.txt` -- every species
  that appears in the reference CSV's own predictions, intersected with the
  model's label set (`src/validation/build_allowlist.py`).
- `models/geomodel_allowlist_218.txt` -- species flagged `in_geo_allowlist=1`
  in `pruning_keepset_493.csv`, the GeoModel side of the keep-set's
  three-source union (`src/pruning/derive_geomodel_allowlist.py`).

Before publishing both side by side, we checked whether they're actually the
same 218 species: **they are identical, set-for-set** -- zero species in
either file that isn't in the other. This is independent confirmation that
the reference CSV's empirically-observed species set really does match the
BirdNET GeoModel's own regional output, previously an inference this project
leaned on rather than a checked fact. Both derivation scripts are in the
repo, so this comparison is reproducible by anyone, not just asserted here.

## Why only one recorder's detail CSV is here

Per-clip detail exists for all six recorders, but the full set is ~28 MB, with
`Audio_Moth_1` alone at 19.8 MB -- most of this repo's size for its
least-read content. Only `Audio_Moth_6` (400 KB, 709 clips) is committed:
it's small, demonstrates the format, and is what
`src/bridge/verify_bridge_parity.py` reads to reproduce the bridge's
bit-exact parity claim independently. The aggregate and per-recorder JSON
files above still carry every headline number for all six recorders.

## Detail CSV columns

| Column | Meaning |
|---|---|
| `clip` | source WAV filename |
| `recorder`, `datetime` | provenance |
| `ref_top1`, `ref_conf` | the birdnet v0.1.7 reference's own prediction |
| `rms` | clip loudness, for context |
| `{fp32,fp16}_raw_top1`, `_raw_conf` | unrestricted argmax over all 493 classes |
| `{fp32,fp16}_allow_top1`, `_allow_conf` | argmax restricted to the 218-species allow-list |
| `{fp32,fp16}_conf_for_ref` | this model's confidence for the reference's own top-1 species |
| `{fp32,fp16}_raw_match`, `_allow_match`, `_conf_match` | agreement booleans against the reference |
| `{fp32,fp16}_det_set`, `_det_n` | detection set at the 0.25 threshold |
| `det_sets_identical` | whether FP32 and FP16 detection sets matched for this clip |
