# BirdNET ONNX Models

Compressed ONNX export of BirdNET v2.4 for the Osa deployment.

**Currently in this folder:** both pruned models — `birdnet_fp16_pruned493.onnx` (18.0 MB)
and `birdnet_fp32_pruned493.onnx` (35.82 MB) — each with its own label file and mapping
CSV, plus `geomodel_allowlist_218.txt` (the GeoModel side of the keep-set's three-source
union; see "What the pruning did" below). The unpruned 6,522-class model is **not**
included here; it is rebuildable from source (see below).

---

## ⚠️ Read this first: every model has exactly one correct label file

This model outputs **numbered slots, not species names**. A separate label file maps slot
number to species, and **this model's mapping is unique to it**.

If you pair it with the wrong label file, **it will load, run, and report the wrong species
with full confidence and no error of any kind.** Nothing throws, nothing looks broken. This
is the single most likely way to get bad results from this repo.

| Model | Outputs | Its ONLY valid label file |
|---|---|---|
| `birdnet_fp16_pruned493.onnx` | 493 | `birdnet_fp16_pruned493.labels.txt` |
| `birdnet_fp32_pruned493.onnx` | 493 | `birdnet_fp32_pruned493.labels.txt` |

Two label files elsewhere in the project are **not** valid for this model:

- **`en_us.txt`** (6,522 entries) belongs to the unpruned model. Index 0 of this pruned
  model is not index 0 of `en_us.txt`; the ordering is entirely different. This is the most
  likely wrong file to reach for, since existing pipeline code loads it by default.
- **`labels.txt`** (6,362 entries) is not valid for anything. It describes a different,
  mislabeled model.

A quick assertion worth putting in any script that loads these:

```python
import onnx
n_out = onnx.load(model_path).graph.output[0].type.tensor_type.shape.dim[-1].dim_value
labels = open(label_path, encoding="utf-8").read().splitlines()
assert len(labels) == n_out, f"label/model mismatch: {len(labels)} labels vs {n_out} outputs"
```

---

## Which model should I use?

**`birdnet_fp16_pruned493.onnx` (18.0 MB)** — 59.4 MB down to 18.0 MB, a 70% reduction,
via two independent steps:

1. **Class pruning**, 6,522 species down to 493. **Exactly lossless** — predictions are
   bit-identical to the full model on every retained species.
2. **FP16 weights.** Very slightly lossy. On 2,000 real field clips it produced an identical
   detection set to FP32 at the 0.25 threshold: same species, same clips, every time.

**Use `birdnet_fp32_pruned493.onnx` (35.82 MB) instead if** you work with raw logits, use a
threshold well below 0.25, or need exact reproducibility against the reference. FP16's
differences live entirely below the detection threshold, but they are not zero.

The unpruned 6,522-class model (for species outside the Osa region) is not included in
this repo; it is rebuildable from source — see below.

---

## What the pruning did

BirdNET's classifier head covers 6,522 species worldwide. The Osa deployment needs a few
hundred. Pruning physically removes the unused output columns from the final layer.

This is **lossless by construction, not approximately**. BirdNET's head is a bare
`MatMul → BiasAdd` with no softmax anywhere in the graph, so each class score is computed
independently from its own weight column. Removing columns cannot perturb the ones that
remain.

**Keep-set (493 species)** — the union of three sources, so that no source's blind spot
can silently drop a species:

```
(Osa masterlist ∩ en_us.txt)  ∪  (species detected in field recordings)  ∪  (GeoModel allow-list)
```

The third source matters: 19 species in the production GeoModel allow-list were missing from
the first two, and pruning without them would have made those species unpredictable.
`*.mapping.csv` records `new_index → original_en_us_index → species` for every retained class.

### Validation

| Check | Result |
|---|---|
| Pruned FP32 vs full FP32, synthetic inputs | `max abs diff = 0.000e+00` |
| Pruned FP32 vs full FP32, 2,000 real clips | `max abs diff = 0.000e+00`, 2011/2011 argmax |
| Allow-list containment | 218/218 present in keep-set |
| **FP16 detection set vs FP32, 2,000 real clips** | **identical, 0 differences at 0.25 threshold** |
| FP16 raw top-1 vs FP32, 2,000 real clips | 99.1%; all disagreements below threshold, mean top-2 margin 0.011 |

Because the pruned model is bit-identical to the full model on retained classes, and the
full model reproduces the `birdnet` v0.1.7 reference at 100.000%, the pruned model's
agreement with the reference follows transitively.

**FP16 note:** ONNX Runtime upcasts FP16 weights to FP32 for compute on CPU, so the FP16
model's accuracy reflects the cost of *storing* weights at half precision, not computing at
it. On hardware with real FP16 units, the cost is unmeasured and could differ.

### Phase 4 six-recorder validation

A separate, larger-scale check than the table above: pruned FP32 against a reference run
of the same model family (`birdnet` v0.1.7), across six field recorders.

| | |
|---|---|
| Clips | 51,720 |
| Allow-list top-1 agreement | **100.000%** |
| Confidence agreement | **100.000%** |
| Largest confidence disagreement anywhere | 1.3e-4 |

100.000% holds on **each of the six recorders individually**, not only in aggregate.

FP16 versus FP32 at the operational 0.25 threshold: identical detection sets on **99.05%**
of clips. Of 497 differing pairs, 460 sit within 0.01 of the threshold boundary —
near-ties rather than substantive disagreements.

**Coverage, stated honestly.** These 51,720 clips are 47.9% of the reference's 108,069
rows. One recorder is complete, another is at 79.6%, and the remaining four are partial.
Per-recorder rates on the smaller recorders rest on a few hundred to a few thousand clips
and should not be read as carrying the same weight as the larger ones.

Full per-recorder numbers: `verification_data/phase4_results_all_recorders.json` and
`../docs/phase4_dual_model_findings.md`. Reproducible from the per-clip detail file for one
recorder (`verification_data/phase4_detail_Audio_Moth_6.csv`) via
`../src/validation/phase4_dual_model.py`.

---

## Rebuilding from source

**The ONNX export cannot be regenerated from the Zenodo source alone, even with exact
version pins.** BirdNET computes its spectrogram with an `RFFT → Cast(real)` pattern that
tf2onnx cannot convert at any opset; conversion fails outright.
[`../src/export/patch_rfft_to_matmul.py`](../src/export/patch_rfft_to_matmul.py) replaces
it with an algebraically exact cosine-DFT matmul. This step is required, not an
optimization.

```
Zenodo BirdNET_v2.4_protobuf.zip (record 15050749, audio-model/)
  → patch_rfft_to_matmul.py          [REQUIRED]
  → tf2onnx (opset 17)               → birdnet_fp32.onnx
  → prune_birdnet_head.py            → birdnet_fp32_pruned493.onnx
  → float16.convert_float_to_float16 → birdnet_fp16_pruned493.onnx   [this file]

Pruning and FP16 conversion commute: doing either order produces bit-identical weights.
```

`en_us.txt` ships inside that same Zenodo archive under `labels/`.

**Environment:** Python 3.11, `onnx==1.18.0` (1.19.0 crashes on import against TF 2.15.1's
`ml_dtypes` cap), `tf2onnx==1.17.0`, `onnxruntime==1.27.0`, `numpy==1.26.4`.

---

## Model I/O



- **Input** `inputs`, shape `(batch, 144000)`, float32 — **raw audio**, 3.000 s at 48 kHz.
  Not a spectrogram; the mel transform happens inside the graph.
- **Output** `scores`, shape `(batch, 493)`, float32 — **raw logits**, not probabilities.
  Input and output stay float32 even though the weights are FP16.

Clips must be **exactly 144,000 samples**. Millisecond-based slicing produces 143,360 or
147,456 and will not match the reference. Apply `flat_sigmoid(sensitivity=-1.0)` to convert
logits to confidences.

---

## Scripts

| Script | Purpose |
|---|---|
| [`../src/pruning/prune_birdnet_head.py`](../src/pruning/prune_birdnet_head.py) | Build a pruned model from any keep-set. Writes model + label file + mapping together, so a model can't be produced without its labels. |
| [`../src/pruning/verify_pruned_model.py`](../src/pruning/verify_pruned_model.py) | Prove a pruned model matches its parent. `--precision-mode` for FP32 vs FP16. |
| [`../src/analysis/diagnose_fp16_flips.py`](../src/analysis/diagnose_fp16_flips.py) | Whether precision differences change any detection above threshold. |
| [`../src/analysis/measure_model_composition.py`](../src/analysis/measure_model_composition.py) | Size breakdown by component, plus the pruning safety gates. |

To prune to a different species list, from `src/pruning/`:

```bash
python prune_birdnet_head.py birdnet_fp32.onnx \
    --keep-labels my_species_list.txt --labels en_us.txt -o my_pruned.onnx
```

The tool refuses to run rather than emit a silently-wrong model. It checks for softmax in
the graph, that the label count matches the model's output width, that head weights aren't
shared with another path, and that the slice is bit-identical before writing.
