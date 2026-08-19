# Class Pruning — Runbook

**Status: Stage 2 (physical slicing) is built, run, and verified.** A working pruned model exists.
What remains is validation against real field audio and the deployment decisions around it.

---

## What's already done

| | |
|---|---|
| `prune_birdnet_head.py` | The pruning tool. Config-driven, four safety gates, writes the label file automatically. |
| `verify_pruned_model.py` | End-to-end verification through real ONNX Runtime inference. |
| `birdnet_fp32_pruned493.onnx` | **Pruned FP32 model. 35.82 MB, down from 59.39 MB — 39.7% smaller.** |
| `birdnet_fp16_pruned493.onnx` | **Pruned + FP16. 18.00 MB — 69.7% smaller.** |
| `birdnet_fp32_pruned493.labels.txt` | 493 labels in pruned output order. **Must ship with the model.** |
| `birdnet_fp32_pruned493.mapping.csv` | `new_index → original_en_us_index → species`. Audit trail. |
| `pruning_keepset_493.csv` | The keep-set with per-species provenance flags (masterlist / field-detected / geo allow-list). |

> **The 474-species build has been retired and deleted.** It failed the allow-list containment check — see below. Anything referencing `*_pruned474.*` is stale.

**Verification result:** bit-identical to the full model on all retained classes, through actual inference, across silence, white noise, full-scale noise, a sine sweep, a pure tone, DC offset, and a near-zero signal. `max |diff| = 0.000e+00`, argmax agreement 11/11, no NaN/Inf anywhere.

Composition after pruning confirms the earlier prediction — the classifier head drops from 42.9% to 5.2%, and **the DFT patch becomes the second-largest component at 28.0%**:

| Component | Before | After |
|---|---|---|
| Conv trunk | 16.43 MB (27.7%) | 16.43 MB (45.9%) |
| **DFT patch** | 10.01 MB (16.9%) | **10.01 MB (28.0%)** |
| POST_CONV | 6.75 MB (11.4%) | 6.75 MB (18.9%) |
| **Classifier head** | **25.50 MB (42.9%)** | **1.85 MB (5.2%)** |

---

## Why masking was skipped as a separate build

The original plan staged this as masking first, then slicing, to de-risk. That staging is no longer the right call, for two reasons found during the audit:

1. **Production already masks.** `run_birdnet.py` applies a 218-species filter *after* inference via `filter_species_by_name`. Building a masking layer would be reimplementing something already running.
2. **Slicing is provably lossless**, so there is nothing for masking to de-risk. The head is a bare `MatMul → BiasAdd` with no softmax, so removing columns cannot perturb retained ones. That's now proven twice — algebraically, and through inference.

**Masking's real value was as an oracle, not a deliverable.** That role is filled: `verify_pruned_model.py` compares the pruned model against the full model's outputs *restricted to the keep-set*, which is exactly the masked model. So the comparison the staging existed to enable is the verification that already ran.

---

## Next steps, in order

### 1. Validate on real field audio *(do this first)*

The synthetic verification proves graph correctness. Running real clips proves it on the actual distribution and rules out anything input-dependent.

```bash
python verify_pruned_model.py \
    birdnet_fp32.onnx \
    birdnet_fp32_pruned493.onnx \
    birdnet_fp32_pruned493.mapping.csv \
    --wav-dir "/path/to/3s48Hz clips" \
    --max-wavs 2000
```

Expect `max |diff| = 0.000e+00` on every clip. **Anything else means stop and investigate** — a nonzero difference here after a clean synthetic run would point at something genuinely surprising, not a rounding issue.

This is cheap (a couple thousand clips is minutes) and it is the gate before anything downstream uses the model.

### 2. ✅ Containment check — RUN, FAILED ON THE 474, FIXED IN THE 493

**This check caught a real defect.** The original 474-species keep-set was missing **19 of the 218 allow-list species**:

`Accipiter striatus` · `Cantorchilus thoracicus` · `Chiroxiphia linearis` · `Crypturellus cinnamomeus` · `Henicorhina leucophrys` · `Leptotila plumbeiceps` · `Manacus candei` · `Nycticorax nycticorax` · `Phaenostictus mcleannani` · `Piranga flava` · `Porphyrio martinica` · `Sayornis nigricans` · `Setophaga palmarum` · `Sporophila corvina` · `Sporophila nigricollis` · `Thryophilus rufalbus` · `Tolmomyias assimilis` · `Trogon collaris` · `Trogon elegans`

These are ordinary Osa-region birds (Collared Trogon, Variable Seedeater, Black Phoebe) that the masterlist's hand-maintained "Included in BirdNET" column simply missed, and that happened not to be detected in the field sample. **Pruning to the 474 would have made all 19 unpredictable**, and re-running Phase 4 would have broken the 100.000% result — reading as a pruning regression when it was really a keep-set gap.

**Root cause: the containment assumption was wrong twice, in the same way.** The original plan assumed the target list sat inside the masterlist's 435; that failed against field data. The revised 474 union assumed the allow-list sat inside masterlist ∪ field-detected; that failed against the GeoModel list. The masterlist's flag column is simply not a reliable upper bound on what BirdNET knows.

**Fix: a three-way union**, now shipped as the 493:

```
keep-set = (masterlist ∩ en_us.txt) ∪ (field-detected) ∪ (GeoModel allow-list)
```

| Source | Count | Unique to it |
|---|---|---|
| Masterlist "Included in BirdNET", resolved | 431 | 275 |
| Field-detected in AudioMoth recordings | 164 | 0 |
| GeoModel allow-list (what production filters on) | 218 | **19** |
| **Union** | **493** | — |

Cost of the fix: **19 species, 0.07 MB.** Containment now passes on both the allow-list and the field-detected set, verified.

A useful consistency check fell out: all 164 field-detected species are inside the 218 allow-list, which they must be, since the reference CSV was generated with that filter applied. That the numbers agree is independent evidence the provenance picture is now correct.

**Standing rule:** any future keep-set change re-runs this check. It is a one-line set difference and it has now caught a defect twice.

### 2b. Re-run Phase 4 scoring against the pruned model

Point `phase4_*.py` at `birdnet_fp32_pruned493.onnx` and `birdnet_fp32_pruned493.labels.txt`. Allow-list top-1 should stay at **100.000%** on the 84,379-clip set. With containment verified there is no longer a known reason for it not to.

### 3. Decide the real keep-set

The 493 is a defensible placeholder, not the answer. Two inputs are still open:

- **KWF's target species list** (Aliyah owns). Drops in as a config file, no code change.
- **The GeoModel list.** Regenerate with `predict_species_at_location_and_time(8.33, -83.3)` and confirm it reproduces the 218. That list is what production actually filters on.

**Recommendation stands: slice to the generous union, keep the geo filter as runtime config.** 493 vs 218 is ~1 MB, and on 4.00% of field clips the unrestricted argmax falls outside the 218 (16.61% on AM4). Slicing tight would convert those into false positives against retained species. Slicing generous costs a megabyte and changes nothing behaviourally.

To build a different keep-set, the tool takes either form:

```bash
# from a resolved CSV
python prune_birdnet_head.py birdnet_fp32.onnx \
    --keep-csv pruning_scaffold_union.csv --labels en_us.txt \
    -o pruned.onnx

# from a plain species list (resolved against en_us.txt, refuses on unresolved names)
python prune_birdnet_head.py birdnet_fp32.onnx \
    --keep-labels reference_csv_species_allowlist.txt --labels en_us.txt \
    -o pruned218.onnx
```

### 4. Stack FP16 on top — DONE, artifact exists

**`birdnet_fp16_pruned493.onnx` — 18.00 MB, a 69.7% reduction from the 59.39 MB original.**

Pruning FP16 works exactly as well as pruning FP32, and the two levers compose cleanly:

| Model | Size | vs. original |
|---|---|---|
| FP32 full | 59.39 MB | — |
| FP16 full | 29.79 MB | 49.8% |
| FP32 pruned (493) | 35.82 MB | 39.7% |
| **FP16 pruned (493)** | **18.00 MB** | **69.7%** |

**Order does not matter — this is provable and was verified.** FP16 conversion is elementwise (rounding, plus clamping of tiny values), and elementwise operations commute with column selection: `round(W)[:, keep] == round(W[:, keep])`. Both orderings were built and the resulting weights are **bit-identical**, as are the file sizes.

`prune_birdnet_head.py` needs no changes to prune an FP16 model — it slices initializers regardless of dtype, and all four gates still pass.

**Slicing remains lossless in FP16.** Pruned-FP16 vs. full-FP16 restricted to the keep-set: `max |diff| = 0.000e+00` across all stress cases, argmax 13/13. So pruning adds no error on top of FP16's error; the two are strictly independent.

**Conventional order is still prune-then-quantize**, and that's what's recommended — not for FP16, where it's mathematically irrelevant, but because it's the robust habit. If INT8 is ever revisited, order matters a great deal there (calibration statistics must reflect the final architecture), and it's better not to have a precedent of doing it backwards.

#### One finding worth knowing: FP16 diverges on digital silence, harmlessly

On an all-zeros input, FP16 and FP32 disagree by up to **8.4 logits** and pick different top-1 classes. That looks alarming and isn't:

- FP32's best logit on silence is **−5.99** (confidence 0.0025); FP16's is **−6.22** (0.0020).
- **Neither produces a single detection above 0.25, or even above 0.10.** Zero detections either way.
- The 1st-vs-2nd logit gap is 0.135, so the argmax is a near-tie among 493 equally implausible classes. There is no meaningful "right answer" to disagree about.

An amplitude sweep shows the effect exists only at *exactly* zero. At 1e-6 and above, the logit difference collapses to ~0.03 and top-1 always agrees:

| Input amplitude | max abs logit diff | top-1 agrees |
|---|---|---|
| 0 (true silence) | 8.4 | no |
| 1e-8 | 0.52 | yes |
| 1e-6 | 0.02 | yes |
| 1e-4 → 1.0 | ~0.03 | yes |

The likely mechanism is the FP16 converter clamping near-zero values to ±1e-7 (it warns about this during conversion, on the DFT cosine matrix entries near cos(π/2)). With no signal to dominate them, those clamped values set the output.

**Practical implication:** harmless for real audio, since real AudioMoth recordings always have some noise floor. But worth two things: don't use digital silence as a parity test case (it will report failures that aren't failures), and **if any clip is ever zero-filled** — the ~75 zero-byte/unreadable clips found in the bundle are a reminder this can happen — FP16 and FP32 will disagree on it while both correctly detecting nothing.

### 5. Then look at the DFT patch

Only worth doing after the above. It's now 28% of the pruned model, it's 426–841× redundant, exactly analytic, and ONNX opset 17 has a native `DFT` op the model could use instead. Removing it projects to **~12.9 MB combined with FP16**, roughly 78% off the original. Needs a bit-parity test to the same standard and an ORT kernel check on the target hardware. Treat as a lead.

---

## The one thing that can silently break this

**The pruned model's outputs are only interpretable through `birdnet_fp32_pruned493.labels.txt`.**

Index 0 of the pruned model is *not* index 0 of `en_us.txt`. If anything downstream pairs the pruned model with `en_us.txt` — or worse, with `labels.txt` — it will run cleanly and report the wrong species. This is the same failure mode the audit caught in the plan, now with a second way to trigger it.

Concretely, for the pack contract:

- The label file is **versioned with the weights**, as one artifact, never separately.
- Nothing downstream hardcodes a class count. It changes from 6,522 to 493 with this model, and will change again when the KWF list lands.
- The mapping CSV should ship too, so any index can be traced back to its original `en_us.txt` position.

`prune_birdnet_head.py` writes all three files together on every run, deliberately — so it is not possible to produce a pruned model without also producing its label file.

---

## Safety gates in the tool

The tool refuses to run rather than produce a silently-wrong model. It checks:

1. **No Softmax in the graph.** If cross-class normalization existed, slicing would change retained scores and the whole approach would be invalid.
2. **Label count == model output width.** This is the `labels.txt` (6,362) vs `en_us.txt` (6,522) trap. Passing the wrong file is caught immediately rather than producing shifted indices.
3. **Head weights consumed by exactly one node each.** If they were shared, slicing could corrupt another path.
4. **Bit-identical slice check** on the weights before writing anything.

Unresolved species are skipped with a printed note (the 3 held for manual review), and the `--keep-labels` path *refuses to run* if any name fails to resolve unless `--allow-unresolved` is passed explicitly.
