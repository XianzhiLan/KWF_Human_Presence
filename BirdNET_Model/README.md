# BirdNET ONNX for the Osa Deployment

ONNX export, class pruning, and validation of BirdNET v2.4 for a rainforest
bioacoustics pipeline running on field-recorded audio from the Osa Peninsula,
Costa Rica. Built for the Kashmir World Foundation's UTSS project, summer 2026.

The short version: BirdNET v2.4 was exported to ONNX, pruned from 6,522 species to
a 493-species regional keep-set, and validated against a reference run at
**100.000% agreement on 51,720 clips across six recorders**. The pruned FP16 model
is 18.00 MB, down from 59.39 MB.

---

## Read this first — three ways to get wrong results with no error

**1. Every model has exactly one correct label file.** These models output numbered
slots, not species names. Pairing a model with the wrong label file produces a
model that loads, runs, and reports the wrong species at full confidence, with
nothing thrown and nothing visibly broken. See [`models/README.md`](models/README.md)
for the pairing table. `labels.txt` (6,362 entries) is a hand-modified artifact and
is valid for **nothing, ever**.

**2. The RFFT patch is required and permanent.** tf2onnx cannot convert BirdNET's
`RFFT → Cast(real)` spectrogram pattern at any opset.
[`src/export/patch_rfft_to_matmul.py`](src/export/patch_rfft_to_matmul.py) replaces
it with an algebraically exact cosine-DFT matrix multiply. **The ONNX artifacts in
this repo cannot be regenerated from the Zenodo source without it**, even with exact
version pins. It costs ~10 MB in the FP32 model.

**3. Clips must be exactly 144,000 samples.** That is 3.0 seconds at 48 kHz, and
288,044 bytes as a WAV. An earlier millisecond-based segmentation produced clips at
147,456 and 143,360 samples; those must not be scored. Tooling here rejects
wrong-length input rather than padding it, because padding produces a plausible
score for a window the reference pipeline never evaluated.

---

## Start here

| If you want to… | Go to |
|---|---|
| Use a model | `models/` — read its README before loading anything |
| See the validation evidence | `docs/phase4_dual_model_findings.md`, `verification_data/` |
| Regenerate the export from scratch | `src/export/`, then `src/pruning/` |
| Prune to a different species list | `src/pruning/prune_birdnet_head.py` |
| Serve the model over HTTP, or benchmark it | `src/bridge/` |
| Know what was measured vs. inherited | `docs/phase4_dual_model_notes.md` |

---

## Models

| File | Size | Classes |
|---|---|---|
| `birdnet_fp16_pruned493.onnx` | 18.00 MB | 493 |
| `birdnet_fp32_pruned493.onnx` | 35.82 MB | 493 |
| original FP32 export | 59.39 MB | 6,522 |

Each pruned model ships with its own `.labels.txt` and a `.mapping.csv` giving
`new_index → original_index → species`.

**Why pruning is lossless, not approximately lossless.** The classification head is
a bare `GlobalAveragePool → Squeeze → MatMul[1024, N] → Add(bias)`. There is no
Softmax anywhere in the graph (verified across all 383 nodes), so each class score
depends only on its own weight column. Removing columns cannot perturb the ones that
remain. Measured difference between pruned and full FP32 on real audio:
`0.000e+00`.

**Keep-set.** 493 species, the union of three sources: the regional masterlist
resolved against `en_us.txt`, species actually detected in field recordings, and the
BirdNET GeoModel's regional allow-list. The union matters — a containment assumption
failed twice during development, and each source contributed species the others
lacked.

**Note on FP16.** ONNX Runtime upcasts FP16 weights to FP32 for computation on CPU
(confirmed by dumping the optimized execution graph: 18 paired cast nodes). The size
reduction is real; latency and memory benefits depend on the deployment target.

---

## Validation

Pruned FP32 against a reference run of the same model family:

| | |
|---|---|
| Clips | 51,720 |
| Allow-list top-1 agreement | **100.000%** |
| Confidence agreement | **100.000%** |
| Largest confidence disagreement anywhere | 1.3e-4 |

100.000% holds on **each of the six recorders individually**, not only in aggregate.

FP16 versus FP32 at the operational 0.25 threshold: identical detection sets on
**99.05%** of clips. Of 497 differing pairs, 460 sit within 0.01 of the threshold
boundary — near-ties rather than substantive disagreements.

The FastAPI service in `src/bridge/` reproduces the validation driver's per-clip
confidences bit-exactly (`0.000e+00`, full float64 precision on both sides with
no rounding in the join) on 709 AM6 clips, for both pruned models. Scope: one
allow-list-winning species per clip, not the full 493-vector, and one recorder —
the bridge inherits the driver's validation rather than making an independent
claim.

**Coverage, stated honestly.** These 51,720 clips are 47.9% of the reference's
108,069 rows. One recorder is complete, another is at 79.6%, and the remaining four
are partial. Per-recorder rates on the smaller recorders
rest on a few hundred to a few thousand clips and should not be read as carrying the
same weight as the larger ones. Per-recorder counts are in
`verification_data/phase4_results_all_recorders.json`.

---

## Environment

Python 3.11. Pins matter here:

```
onnx==1.18.0          # NOT 1.19.0
tf2onnx==1.17.0
onnxruntime==1.27.0
numpy==1.26.4
```

Models are IR version 8, opset 17.

Preprocessing, proven bit-exact against the reference implementation
(`0.000e+00`): read as float32 → resample to 48 kHz → split into 144,000-sample
windows → no-op bandpass → model → `flat_sigmoid(sensitivity=-1.0)` on the output
logits. The mel transform happens inside the graph; the model takes raw audio.

---

## Repository layout

```
models/              ONNX models, each with its label and mapping files
src/
  export/            RFFT patch — required to regenerate the ONNX artifacts
  pruning/           keep-set construction, pruning, and the verification oracle
  validation/        inventory, dual-model scoring, selective archive extraction
  analysis/          model size composition, FP16 flip diagnosis
  bridge/            FastAPI service for serving and benchmarking the models
scripts/             original pipeline scripts (author: Qian Fu)
verification_data/   Phase 4 results, per-clip detail, keep-set with provenance
docs/                pruning plan, runbook, findings, and status write-ups
notebooks/           exploratory analysis
```

`docs/pruning_runbook.md` includes the containment-failure record: what was assumed,
how it failed, and the one-line check that has now caught the same class of defect
twice. It is probably more useful than any of the code here.

---

## What is not in this repo

- **Audio.** ~39 GB of field recordings, not ours to publish.
- **The AED and human-presence models.** BirdNET is one of three models in the
  deployment chain; this repo covers BirdNET only.

**Present, and flagged rather than hidden:** the reference CSV
(`verification_data/birdnet_species_108k_fp16_verification.csv`) **is** in
this repo, committed in an earlier commit. Its `audio_path` column contains
absolute filesystem paths from a collaborator's machine. It is not removed
here — that's a separate decision from a README edit — but anyone auditing
this repo for exposure should be told it's there, not told it isn't.

---

## Credits

Original pipeline scripts in `scripts/` by Qian Fu. ONNX export, pruning,
validation, and bridge by Coleman Bryant. BirdNET v2.4 is by the Cornell Lab of
Ornithology and Chemnitz University of Technology
([Zenodo record 15050749](https://zenodo.org/records/15050749)); this repository
contains a derived export, not the original model.
