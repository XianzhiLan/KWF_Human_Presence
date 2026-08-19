# BirdNET ONNX Export — Status Update

**Author:** Coleman
**Date:** 2026-07-06
**Scope:** BirdNET FP32 ONNX export, and a data-segmentation issue found while validating it against the 108k reference CSV.

---

## Summary

The BirdNET FP32 ONNX export is **built and validated**. On every clip I could correctly match against the reference CSV, the ONNX model reproduces `birdnet` v0.1.7 exactly: **37,596 clips, 100.000% top-1 agreement, matching confidences.**

While validating, I found that most of the shared audio is segmented to a slightly different clip length than the reference CSV was built from, so it can't be scored against that CSV yet. This is a data/segmentation issue on the audio side, not a problem with the model or the conversion. Details and what would unblock a full run are below.

---

## What's validated

The FP32 ONNX model is a faithful port of `birdnet` v0.1.7:

- **37,596 correctly-segmented clips scored, 100.000% agreement**, 100% confidence-match (36,919 from Audio_Moth_1 plus 677 correct-length Audio_Moth_2 clips). Zero mismatches.
- No accuracy loss on quiet/low-signal clips (a specific thing I'd been watching, since the export required a small internal graph change to the spectrogram computation — that change is exact, and this confirms it holds on real audio).
- The preprocessing that feeds the model matches `birdnet`'s exactly (bit-for-bit), so the comparison is apples-to-apples.
- Cross-checked against the original TensorFlow model directly: on the same audio, the ONNX export and the original model agree to within rounding, confirming the port didn't change behavior.

Bottom line: the export can be trusted, and the compression work (FP16, class pruning) can build on it.

---

## The data-segmentation issue

**What I found:** the reference CSV (`birdnet_species_108k_fp16_verification.csv`) was generated from audio cut to exactly **144,000 samples (3.000 s @ 48 kHz)**. Most of the shared clips are cut slightly differently — **143,360 samples (~2.987 s)** or **147,456 samples (~3.072 s)** — which looks like a different segmentation run. Clips at the wrong length don't line up with their CSV row, so they can't be validated against it.

**How I know it's segmentation, not the model:** the clips still produce correct, sensible predictions — they just don't match a CSV that was built from differently-cut audio. Confirmed three independent ways (bit-exact preprocessing, original TF model agreeing with the ONNX export on the same files, and the fact that the mismatched clips' predictions don't appear anywhere in the CSV — so it's a different cut of the same recordings, not a labeling or indexing bug).

**A note on completeness:** an earlier read suggested a lot of clips were missing from the share. That turned out to be a partial extraction on my end — the full `all_filtered_clips.zip` bundle actually contains essentially all the referenced clips by name (98,631 of 108,069). The real issue is length/segmentation, not missing files. The one genuine exception is Audio_Moth_1 (see the table).

---

## Per-recorder status

| Recorder | Clips referenced by CSV | Present in bundle | At correct length (144k) | Status |
|---|---|---|---|---|
| Audio_Moth_1 | 46,360 | 36,922 | 36,922 (100% of present) | **Validated.** Correct length. 9,438 clips absent from the bundle. |
| Audio_Moth_2 | 24,493 | 24,493 | 2,027 (8.2%) | Needs re-segmentation. |
| Audio_Moth_3 | 2,575 | 2,575 | 0 | Needs re-segmentation. |
| Audio_Moth_4 | 7,358 | 7,358 | 0 | Needs re-segmentation. |
| Audio_Moth_5 | 13,541 | 13,541 | 0 | Needs re-segmentation. |
| Audio_Moth_6 | 13,742 | 13,742 | 0 | Needs re-segmentation. |

Audio_Moth_1 is the only recorder mostly at the correct length, which is why the validation rests on it (plus the small AM2 correct-length subset).

---

## What would unblock a full multi-recorder validation

1. **Re-segment Audio_Moth_2 through Audio_Moth_6 at exactly 144,000 samples (3.000 s @ 48 kHz)** — the same segmentation that produced the reference CSV — *if that's feasible.* The key question is whether the original continuous recordings still exist: re-cutting from those to a 144,000-sample window is straightforward, but if only the current (shorter) clips remain, the missing samples can't be recovered and we'd need another approach. The bundle already has all these clips by name, just at the wrong length, so re-sharing the current bundle won't help. (AM2 is included here — only 8.2% of it is correct-length, so it has the same issue as AM3–AM6.)

2. **Audio_Moth_1 completeness:** the bundle holds 36,922 of the 46,360 AM1 clips the CSV references — 9,438 are absent. The `clips_1` parts look like a complete download sequence, so the folder itself may just be short rather than a failed upload. Worth confirming whether `clips_1` is complete.

3. **Segmentation pipeline check (optional, diagnostic):** worth a look at why some sessions land at 2.987 s / 3.072 s instead of a clean 3.000 s — possibly rounding on start/stop sample indices, or overlap handling. This is the root cause of the length mismatch, so fixing it at the source would prevent it recurring.

4. **Minor:** ~75 zero-byte/unreadable clips across recorders, and one AM6 clip truncated to ~1 s, are present in the bundle — worth regenerating alongside the rest.

Once corrected clips are in place, re-running is cheap — the scoring pipeline generalizes to more recorders with no changes.

---

## What's next on my end (not blocked by the above)

The export is validated on Audio_Moth_1, so the compression work proceeds in parallel:

- **FP16 conversion and agreement baseline** — comparing the compressed model against the validated FP32 export directly (which sidesteps the segmentation issue entirely, since both are our own models on the same clips).
- **System metrics** (size, latency, memory) for the digital-twin evaluation.
- **Class pruning** toward the KWF target-species list.

None of these are blocked by the recorder re-export — that only affects how broadly the FP32 validation can be extended across recorders, which is a robustness nice-to-have on top of an already-solid result.

---

## Reference (for anyone picking this up later)

- **Model source:** BirdNET v2.4 SavedModel, Zenodo record 15050749.
- **Reference predictions:** `birdnet_species_108k_fp16_verification.csv`, generated by `birdnet` v0.1.7 (FP32 output; the "fp16" in the filename refers to what it was used to verify, not its own precision). Built against a ~218-species regional allow-list, not the model's full 6,522-class output — relevant when comparing any future model variant against it.
- **One export caveat worth recording:** the SavedModel → ONNX conversion required a small, exact graph change to BirdNET's spectrogram computation (its internal FFT pattern isn't directly convertible by the standard tooling; it was replaced with a mathematically-identical matrix operation, verified against the original model). Anyone regenerating the ONNX export from source needs to apply that step — it's documented in the export repo.
