# Phase 4 recorder-coverage extension — FP32 and FP16 pruned-493

Every CSV-referenced clip currently on disk, scored under both pruned 493-class models
in a single decode-once pass. **51,720 clips across all six recorders**, run in two
passes: Pass A (AM1, AM2, AM4, AM6) and Pass B (AM3, AM5) once their archives finished
downloading. Pass B results were appended to Pass A rather than recomputing it.

Artifacts: `phase4_results_all_recorders.json`, `phase4_results_{recorder}.json`,
`phase4_detail_{recorder}.csv`, `phase4_negative_check.json`, `preflight_inventory.json`.
Assumptions and what was verified versus inherited: `phase4_dual_model_notes.md`.

## Headline

**The FP32 pruned-493 model reproduces the reference exactly on all 51,720 clips** —
100.000% allow-list top-1 *and* 100.000% confidence-match on every one of the six
recorders individually, not merely in aggregate. The largest confidence disagreement
anywhere in the run is 1.3 × 10⁻⁴, with a median of 1.4 × 10⁻⁶, and not a single clip
exceeds the 1 × 10⁻³ tolerance.

**FP16 costs 66 top-1 flips out of 51,720 (99.872% agreement)**, and 58 of the 66 swap
one nocturnal owl for another at a median margin of 2.3 × 10⁻³.

## Stage 1 — inventory

| recorder | files | correct_len | bad_147456 | bad_143360 | other | csv_rows_ondisk | scoreable |
|---|---|---|---|---|---|---|---|
| Audio_Moth_1 | 37,079 | 37,068 | 0 | 0 | 11 | 36,919 | 36,919 |
| Audio_Moth_2 | 9,360 | 9,360 | 0 | 0 | 0 | 2,026 | 2,026 |
| Audio_Moth_3 | 3,208 | 3,208 | 0 | 0 | 0 | 1,208 | 1,208 |
| Audio_Moth_4 | 118,665 | 118,665 | 0 | 0 | 0 | 7,358 | 7,358 |
| Audio_Moth_5 | 5,500 | 5,500 | 0 | 0 | 0 | 3,500 | 3,500 |
| Audio_Moth_6 | 8,331 | 8,331 | 0 | 0 | 0 | 709 | 709 |

Not a single file at 295,170 or 286,978 bytes anywhere in the tree — the pre-fix
segmentation is fully gone, so no recorder was excluded for contamination. The 11
`other` files are zero-byte AM1 artifacts and were dropped. No basename resolved to two
paths, including across the Pass B directories.

## Stage 2 — gate

500 clips from Audio_Moth_6, the smallest passing recorder. FP32 allow-list top-1 came
in at 100.000% on the first attempt, so the run proceeded. All pre-inference assertions
passed: both label files exactly 493 entries and byte-identical (SHA256 `72b910ab…`),
neither is `en_us.txt` or `labels.txt`, both models emit 493, and the allow-list mask
built by name against the 493 space sums to exactly 218.

## Stages 3 and 4 — per-recorder results

**FP32**

| recorder | date coverage | clips | allow-list top-1 | conf-match | raw top-1 |
|---|---|---|---|---|---|
| Audio_Moth_6 | 2025-03-21 (1 date) | 709 | 100.000% | 100.000% | 100.000% |
| Audio_Moth_3 | 2025-03-19..20 (2 dates) | 1,208 | 100.000% | 100.000% | 100.000% |
| Audio_Moth_2 | 2025-03-17 (1 date) | 2,026 | 100.000% | 100.000% | 99.901% |
| Audio_Moth_5 | 2025-03-19..20 (2 dates) | 3,500 | 100.000% | 100.000% | 92.400% |
| Audio_Moth_4 | 2025-03-17..21 (5 dates) | 7,358 | 100.000% | 100.000% | 85.947% |
| Audio_Moth_1 | 2025-03-17..20 (4 dates) | 36,919 | 100.000% | 100.000% | 98.499% |
| **TOTAL** | | **51,720** | **100.000%** | **100.000%** | 96.411% |

**FP16**

| recorder | date coverage | clips | allow-list top-1 | conf-match | raw top-1 |
|---|---|---|---|---|---|
| Audio_Moth_6 | 2025-03-21 (1 date) | 709 | 100.000% | 36.953% | 100.000% |
| Audio_Moth_3 | 2025-03-19..20 (2 dates) | 1,208 | 100.000% | 46.689% | 100.000% |
| Audio_Moth_2 | 2025-03-17 (1 date) | 2,026 | 99.951% | 61.106% | 99.852% |
| Audio_Moth_5 | 2025-03-19..20 (2 dates) | 3,500 | 99.743% | 27.029% | 92.371% |
| Audio_Moth_4 | 2025-03-17..21 (5 dates) | 7,358 | 99.973% | 39.481% | 85.906% |
| Audio_Moth_1 | 2025-03-17..20 (4 dates) | 36,919 | 99.854% | 36.515% | 98.380% |
| **TOTAL** | | **51,720** | **99.872%** | 37.502% | 96.317% |

Raw top-1 is the unrestricted argmax over all 493 pruned classes measured against a
reference whose own predictions were filtered to the 218 allow-list species. The two
quantities are not measuring the same thing, and AM4's 85.9% reflects how often its
global argmax lands on one of the 275 non-allow-list species rather than any error.

### The FP16 conf-match number is a tolerance artifact, not a precision problem

37.5% conf-match looks alarming next to 99.872% top-1 agreement. The two measure very
different things. `CONF_TOL` is 1 × 10⁻³, inherited unchanged from the validated drivers,
and FP16's median confidence deviation is 1.46 × 10⁻³ — sitting just above the line, so
roughly 62% of clips fail a threshold they miss by a hair.

Widening the lens by one decimal place:

| deviation from reference confidence | FP32 | FP16 |
|---|---|---|
| median | 1.4 × 10⁻⁶ | 1.5 × 10⁻³ |
| maximum | 1.3 × 10⁻⁴ | 1.9 × 10⁻¹ |
| within 1 × 10⁻³ | 100.000% | 37.50% |
| within 1 × 10⁻² | 100.000% | 98.670% |
| within 5 × 10⁻² | 100.000% | 99.901% |

FP16 confidences agree with the reference to two decimal places on 98.67% of clips. The
37.5% figure is the right answer to "does FP16 reproduce the reference to 1e-3" — and the
answer is no — but it is the wrong number to quote as an accuracy measure.

### The 66 FP16 flips are near-ties among confusable owls

58 of 66 (87.9%) swap one nocturnal owl for another: Black-and-white Owl, Crested Owl,
Mottled Owl, and Spectacled Owl trading places among themselves. The median margin
between winning and displaced species at the moment of the flip is 2.3 × 10⁻³. These are
coin-flips that FP16 rounding tipped, concentrated in a genuinely confusable acoustic
group recorded at night — the benign pattern, not a systematic FP16 failure mode.

### One class is disproportionately FP16-sensitive

The largest FP32-vs-FP16 confidence deviation in the run is 1.904 × 10⁻¹, on
`Audio_Moth_5_20250320_052102.wav` for **Black-cheeked Ant-Tanager**
(*Habia atrimaxillaris*). That same species also held the maximum in Pass A before AM5
existed (5.3 × 10⁻² on AM4). Being the top deviant twice, on two different recorders,
makes it worth watching rather than dismissing as a one-off: whatever weight path drives
this class appears more exposed to FP16 rounding than the other 492.

## Stage 5 — FP16 vs FP32 detection sets at 0.25

The 0.25 threshold was specified independently of the data, and it turns out to be
exactly the reference CSV's own inclusion floor: the minimum confidence across all
108,069 rows is 0.250000, on every recorder, with zero rows below. The comparison is
therefore happening at the boundary that actually matters.

- Clips compared: 51,720
- Identical detection sets: 51,228 (**99.0487%**)
- Clips differing: 492, comprising 497 (clip, species) pairs
- Direction: 360 dropped by FP16, 137 added by FP16
- **460 of the 497 pairs have both confidences within 0.01 of the 0.25 boundary**
- Max |FP32 − FP16| deviation across all 493 classes and all clips: 1.904 × 10⁻¹, at
  `Audio_Moth_5_20250320_052102.wav`, Black-cheeked Ant-Tanager
- Closest approach to the boundary by any class in either model: 8.941 × 10⁻⁸, at
  `Audio_Moth_1_20250319_024851.wav`, Mottled Owl, FP16, confidence 0.249999911

Nearly every set difference is a species sitting on the threshold that FP16 nudged
across. Since the closest approach to the boundary (8.9 × 10⁻⁸) is far smaller than the
typical FP16 deviation, set differences at this scale are expected arithmetic, not a
defect.

## Stage 6 — negative check on non-CSV clips

Seed 20260809, up to 2,000 correct-length non-CSV clips per recorder, both models, same
0.25 threshold. Because the CSV's floor is verifiably 0.25, a clip absent from the CSV
should carry no allow-list detection at or above it.

| recorder | pool | sampled | FP32 detections | notes |
|---|---|---|---|---|
| Audio_Moth_2 | 7,334 | 2,000 | 0 | one FP16-only hit at 0.2518, a pure boundary straddle |
| Audio_Moth_3 | 2,000 | 2,000 | 0 | clean |
| Audio_Moth_4 | 111,307 | 2,000 | 1 | single clip at 0.4193 |
| Audio_Moth_5 | 2,000 | 2,000 | 2 | median 0.436, max 0.533 |
| Audio_Moth_6 | 7,622 | 2,000 | 13 | median 0.346, 4 at or above 0.5 |
| Audio_Moth_1 | 149 | 149 (pool exhausted) | 155 across all 149 clips | median 0.470, 74 at or above 0.5 |

Five of six recorders behave as expected, at 0 to 13 detections per 2,000 clips.

**Audio_Moth_1 is the finding here, and it is a data-provenance result rather than a
model result.** Every one of its 149 non-CSV clips produced an allow-list detection, at
a median confidence of 0.470 and a maximum of 0.996. Those are not marginal calls. The
straightforward reading is that these 149 clips should have CSV rows and do not — a gap
in the reference, not a false-positive tendency in the model. I checked whether they were
duplicate-download artifacts, and only 2 of the 149 carry a `(N)` suffix, so that does
not explain it. Their timestamps fall inside dates the CSV already covers for AM1, which
rules out a whole-session omission.

This also qualifies the negative check itself: it assumes the reference pipeline was run
over every clip on disk, and AM1 shows that assumption does not universally hold. The
other five recorders' numbers are meaningful; AM1's is measuring the reference's coverage,
not the model's specificity.

## Coverage — what fraction of the reference this actually covers

51,720 of the 108,069 CSV rows, or **47.9%**. Per recorder:

| recorder | CSV rows | scored | coverage |
|---|---|---|---|
| Audio_Moth_1 | 46,360 | 36,919 | 79.6% |
| Audio_Moth_2 | 24,493 | 2,026 | 8.3% |
| Audio_Moth_3 | 2,575 | 1,208 | 46.9% |
| Audio_Moth_4 | 7,358 | 7,358 | 100.0% |
| Audio_Moth_5 | 13,541 | 3,500 | 25.8% |
| Audio_Moth_6 | 13,742 | 709 | 5.2% |

The Pass B archives are single-date exports (`Audio_Moth_3_031925`, `Audio_Moth_5_031925`)
covering 2025-03-19 and part of 03-20, which is why AM3 and AM5 land well short of their
full CSV row counts despite extracting every matching member the archives contained.

## What this does and does not establish

Established: the FP32 pruned-493 export is exact against the reference on every clip we
can currently check — all six recorders, five dates, 51,720 clips, zero disagreements in
either top-1 or confidence. FP16 costs 0.128% of top-1 decisions, with the damage
concentrated in near-ties among confusable owls.

Not established: AM2 and AM6 contribute only 8.3% and 5.2% of their CSV rows, so their
per-recorder percentages rest on a small and possibly unrepresentative slice of those
sessions; AM6's 100.000% FP16 agreement in particular is a 709-clip result. Over half the
reference remains unscored for want of audio. All FP16 figures reflect ONNX Runtime CPU
behavior, which stores weights in FP16 but computes in FP32 via upcast; true FP16-compute
hardware is untested and remains the open question recorded in the Phase 5 findings.
