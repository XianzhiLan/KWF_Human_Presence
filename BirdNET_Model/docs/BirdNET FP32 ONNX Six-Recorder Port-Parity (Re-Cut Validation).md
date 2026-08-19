# BirdNET FP32 ONNX — Six-Recorder Port-Parity (Re-Cut Validation) + Data-Coverage Findings

Scope: validate the FP32 ONNX export against the reference CSV
(`birdnet_species_108k_fp16_verification.csv`) across **all six AudioMoth recorders**, using
Qian's **re-cut, sample-accurate 144000-sample clips**. Result: the export reproduces the
reference **exactly (100.000% allow-list top-1) on every recorder**. The clip-length problem
that previously blocked AM2–AM6 is **fully resolved**. The only remaining issue is **data
coverage** — a large fraction of the CSV-referenced clips are not yet on disk, in
recorder-specific contiguous time windows. No Phase 5 (FP16) work is in scope here.

## TL;DR

- **The model/conversion is fully validated across all six recorders.** On correctly-segmented
  (144000-sample / exactly 3.0 s) clips, the FP32 ONNX model reproduces the birdnet v0.1.7
  reference at **100.000% allow-list top-1 and 100.000% confidence-match**, on **60,243 clips
  total** spanning AM1–AM6. Zero allow-list mismatches, zero quiet-clip regressions.
- **The re-cut fixed the length problem.** Every re-cut clip on disk (AM2–AM6) is now exactly
  144000 samples — 0 off-length, 0 unreadable. The old 143360 / 147456-sample segmentations
  (and AM4's associated 2-chunks-per-clip join ambiguity) are gone.
- **Remaining issue is coverage, not correctness.** Across AM2–AM6, **38,385 CSV-referenced
  clips are still absent from disk**. The absences are **contiguous time windows** (whole days
  / clean hour boundaries), i.e. a source/extraction coverage gap — **not** random sync loss
  and **not** a model problem. Re-syncing won't recover them; those spans need re-export.

## Headline results — all six recorders (144000-sample scope)

Confidence uses `flat_sigmoid(sensitivity=-1.0)` (birdnet's effective default), matching the
CSV. "allow-list top-1" restricts our argmax to the 218 species the CSV was generated with.

| recorder | clips scored | allow-list top-1 | conf-match | raw top-1 | quiet-decile top-1 |
|---|---|---|---|---|---|
| Audio_Moth_1 | 36,919 | **100.000%** | 100.000% | 97.71% | 100.000% |
| Audio_Moth_2 | 13,736 | **100.000%** | 100.000% | 95.49% | 100.000% |
| Audio_Moth_3 | 1,367 | **100.000%** | 100.000% | 96.34% | 100.000% |
| Audio_Moth_4 | 3,866 | **100.000%** | 100.000% | 83.39% | 100.000% |
| Audio_Moth_5 | 3,646 | **100.000%** | 100.000% | 95.91% | 100.000% |
| Audio_Moth_6 | 709 | **100.000%** | 100.000% | 98.87% | 100.000% |
| **Total** | **60,243** | **100.000%** | **100.000%** | — | **100.000%** |

Every recorder: **0 allow-list top-1 mismatches**. Raw top-1 (unrestricted 6,522-class argmax)
varies by recorder because our model sometimes wins with a species outside the CSV's
218-species filtered universe; the correct comparison (the CSV was generated under that
allow-list) is allow-list top-1 = 100%, corroborated by conf-match = 100%.

AM1 was scored from the Drive zip's correct-length set; AM2–AM6 from Qian's re-cut
`3s48Hz clips` folder.

## Re-cut length check (the fix, confirmed)

Header scan of every re-cut clip on disk (48 kHz / mono / PCM_16):

| recorder | on disk (wav) | @144000 (correct) | off-length | unreadable |
|---|---|---|---|---|
| Audio_Moth_2 | 66,596 | 66,596 (100%) | 0 | 0 |
| Audio_Moth_3 | 28,798 | 28,798 (100%) | 0 | 0 |
| Audio_Moth_4 | 57,573 | 57,573 (100%) | 0 | 0 |
| Audio_Moth_5 | 37,671 | 37,671 (100%) | 0 | 0 |
| Audio_Moth_6 | 8,331 | 8,331 (100%) | 0 | 0 |

Phase 3 preprocessing-parity gate re-run on a sample spanning each recorder:
`max_abs_diff = 0.000e+00`, all single-chunk — bit-exact to birdnet v0.1.7 on the re-cut audio.

## What was validated (unchanged; the model was never the problem)

- **Preprocessing is bit-exact.** birdnet v0.1.7's real load path vs our reimplementation:
  `max_abs_diff = 0.000e+00` on every sampled clip, all six recorders.
- **ONNX == original TensorFlow.** On identical on-disk audio the unpatched
  `BirdNET_v2.4_protobuf/audio-model` SavedModel and `birdnet_fp32.onnx` agree to ~1e-4 logits
  and produce identical top-1.
- **Label indexing confirmed.** `en_us.txt` = 6,522 lines, no duplicates; birdnet's
  `OrderedSet(splitlines())` collapses nothing → plain 1:1 index map, matched on our side.
- **Allow-list resolved.** The reference CSV is a filtered run over **218 distinct species**
  (all valid `en_us.txt` labels; some cells carry 2 species). Persisted to
  `reference_csv_species_allowlist.txt`.

## Coverage: CSV-referenced clips present vs absent

The re-cut folder holds many clips beyond the CSV's filtered subset, but of the specific clips
the CSV references, a large fraction is not yet on disk:

| recorder | CSV-referenced | present on disk | absent | absent % |
|---|---|---|---|---|
| Audio_Moth_2 | 24,493 | 13,736 | 10,757 | 43.9% |
| Audio_Moth_3 | 2,575 | 1,367 | 1,208 | 46.9% |
| Audio_Moth_4 | 7,358 | 3,866 | 3,492 | 47.5% |
| Audio_Moth_5 | 13,541 | 3,646 | 9,895 | 73.1% |
| Audio_Moth_6 | 13,742 | 709 | 13,033 | 94.8% |
| **AM2–AM6 total** | **61,709** | **23,324** | **38,385** | 62.2% |
| Audio_Moth_1 (from zip) | 46,360 | 36,919 | 9,438 | 20.4% |

Per-recorder absent-clip filenames are saved to `missing_am2_recut.txt` … `missing_am6_recut.txt`.

**The gaps are contiguous time windows, not scattered files.** Hour-by-hour, present hours are
~100% present and missing hours ~100% missing (only the boundary hour splits) — the signature
of session/coverage truncation, not random per-file transfer loss. Covered windows by recorder:

- **AM2** — missing 2025-03-17 18:00 → 2025-03-19 06:00; covered 03-17 day, 03-19 day, 03-20, 03-21.
- **AM3** — missing all of 03-19 + 03-20 pre-06:00; covered 03-20 07:00 → 03-21.
- **AM4** — missing 03-17, 03-18, 03-19 pre-06:00 (leading) and 03-21 post-06:00 (trailing);
  covered 03-19 07:00 → 03-21 06:00.
- **AM5** — missing 03-17, 03-18, 03-19 entirely + 03-20 partial; covered ~03-20 midday → 03-21.
- **AM6** — missing 03-17 through 03-20 entirely; covered only part of 03-21 (709 clips).

Coverage worsens monotonically by recorder number (AM2 44% → AM6 95% absent), consistent with
the extraction/re-cut covering progressively less of each recorder's timeline.

## Background: the original root cause (now fixed)

The reference CSV was generated from audio cut to **exactly 144000 samples**. The previously
shared bundle (`all_filtered_clips.zip`) contained AM2–AM6 clips by name but segmented to
**143360** (≈2.987 s) or **147456** (≈3.072 s) samples — a `pydub` millisecond-based
segmentation script rounding start/stop indices, i.e. a **different segmentation run** from the
one that produced the CSV. That is why same-named clips previously failed to match. Qian
re-cut the audio with **sample-accurate slicing at exactly 144000 samples**, which this run
confirms resolves the mismatch on all six recorders. (This supersedes the earlier note's
"re-export at 144000" ask, which is now complete.)

## Remaining asks (coverage only)

1. **Re-cut method is confirmed correct — no further segmentation work needed.** All six
   recorders validate at 100% on the sample-accurate 144000-sample clips.
2. **Fill the coverage gaps.** ~38k CSV-referenced clips (AM2–AM6) are not yet on disk, in the
   contiguous windows listed above (per-clip lists in `missing_am*_recut.txt`). Highest
   priority: **AM5 (73% absent) and AM6 (95% absent)**, whose present sets are small and skewed
   to a single date. These are source-coverage gaps — re-syncing won't help; the missing spans
   need to be re-extracted/exported.
3. **Confirm whether those windows exist at source.** For each recorder the gap is a specific
   date/time range, not random dropouts — worth checking whether that audio was recorded and
   simply not included in this extraction batch, vs. never captured.
4. **AM1 completeness (unchanged):** the bundle holds 36,919 of the 46,360 AM1 clips the CSV
   references — 9,438 absent. Please confirm whether the clips_1 source is complete.

Re-running once corrected clips land is cheap: the per-recorder drivers below are thin clones
of the validated AM1 methodology; the Phase 3 gate and allow-list are unchanged.

## Artifacts produced

Shared / validated:
- `birdnet_preprocessing.py` — preprocessing/label/sigmoid module (bit-exact to birdnet v0.1.7)
- `reference_csv_species_allowlist.txt` — 218-species allow-list
- `birdnet_fp32.onnx` — the validated FP32 ONNX export

Per-recorder re-cut validation (this run; new thin drivers, no validated scripts edited):
- `diagnose_am{2..6}_recut.py` — length check + CSV cross-reference → `missing_am{2..6}_recut.txt`
- `phase3_parity_recut.py`, `phase3_parity_am{4,5,6}.py` — Phase 3 gates (all `0.000e+00`)
- `phase4_am2_am3_recut.py` → `phase4_am2_am3_recut_results.csv`
- `phase4_am{4,5,6}_recut.py` → `phase4_am{4,5,6}_recut_results.csv`

AM1 full correct-length re-score (from Drive zip):
- `phase4_am1_full.py` → `phase4_am1_full_results.csv` (36,919 clips, 100.000% allow-list top-1)

Earlier diagnostics (superseded by the re-cut, kept for provenance):
- `phase4_am1_am2.py`, `diagnose_am2.py`, `diagnose_am36.py`, and the zip-reconciliation scan.
