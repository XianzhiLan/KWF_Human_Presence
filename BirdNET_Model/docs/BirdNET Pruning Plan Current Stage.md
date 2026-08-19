> ⚠️ **Superseded — retained for historical record.** This plan predates three
> corrections: the canonical label file is `en_us.txt` (6,522 classes), not
> `labels.txt` (6,362); the final keep-set is 493 species from a three-way union,
> not the 435-species scaffold, because the containment assumption asserted below
> failed twice; and the masking stage was deliberately skipped in favour of
> slicing directly. See `docs/pruning_runbook.md` for what actually happened and
> `docs/phase4_dual_model_findings.md` for current results.

# BirdNET Class Pruning — Plan

## Core approach

Class pruning is the lead compression strategy: slice BirdNET's 6,362-class output layer down to only the species the deployment needs. It operates on the already-trained model's final layer, so it is **training-free** — no fine-tuning, no retraining.

That property is the whole reason it's viable. QAT and weight pruning are off the table because they require Cornell's BirdNET training pipeline and dataset, which we don't have. Class pruning needs neither. It also gives:

- **Zero fidelity loss on retained species** — the kept classes are untouched, so their confidences are identical to the full model.
- **False-positive reduction** — species that can't occur in the deployment region can no longer be predicted at all.

## One-line summary

Training-free output-layer class pruning, built mask-first then slice, configurable around the Osa ~435 scaffold, with a mandatory label-index parity check, and the slicing stage gated on the ONNX FP32 export -- Likely my first main task, unless it has already been done.

## Two-stage build

Build the mechanism in two stages, simplest first.

**Stage 1 — Masking.** Keep the full 6,362-wide model and zero/ignore every output outside the keep-set. Runtime-agnostic (works on TFLite or ONNX), reversible, and enough to measure false-positive reduction and retained-species agreement. This is the early, low-risk version.

**Stage 2 — Physical slicing.** Actually excise the output-layer rows for dropped classes, shrinking the weight matrix. This is where the real size and compute win comes from. It wants the ONNX graph to operate on (clean, well-tooled graph surgery via the `onnx` Python library), unlike the TFLite flatbuffer which is painful to edit in place.

Start with masking to de-risk and measure; move to slicing for the actual model-size reduction once the ONNX FP32 export exists and the approach is validated.

## Configurable around a placeholder set

Build the mechanism against the **Osa-in-BirdNET ~435 set** (the masterlist's "Included in BirdNET" species) as a *safe upper-bound scaffold*. Containment is the justification: the real KWF target list almost certainly sits inside the 435, since any usable species must be both regionally present and BirdNET-recognizable — the two filters that define the 435.

Rules for staying configurable:
- The keep-set is a **config input** (file or config entry the pruning step reads), never a hardcoded array. The HP notebooks' failure mode was inline species lists drifting out of sync across cells; this is the inverse discipline.
- **Don't tune anything to the specific composition of 435** — no thresholds or per-class calibration that assume those exact species. Treat 435 as a placeholder of roughly the right size; optimize the machinery, not the list.
- The real target list drops in later as a parameter, no code change.

## Label-parity guard (required before any cut)

Every keep-set species string must resolve to **exactly one** BirdNET output index, matching `labels.txt` order exactly. Verify this before slicing or masking anything. A list that *looks* aligned but is offset by mismatched ordering is the same class of bug as the FP16 label-index misalignment from the benchmark audit — it produces a model that loads fine but predicts the wrong species. One verification pass: confirm each of the keep-set strings maps to a unique, correct label index.

## Dependencies and sequencing

| Piece | Can start | Blocked on |
|---|---|---|
| Masking mechanism + label-parity check | **Now** | only `labels.txt` + Osa masterlist |
| Measuring masking (FP reduction, retained-species agreement) | After | audio subset + verified ONNX FP32 baseline |
| Physical slicing (real size/compute win) | After | verified ONNX FP32 export; informed by Kyle's metric answer |

The masking mechanism runs in **parallel** with the ONNX FP32 export task — it doesn't need the model running. But *measuring* and *shipping* pruning both wait on the verified ONNX FP32 baseline, so you have a clean reference to attribute changes against.

## Inputs still needed

- **KWF target-species list** — needed to *finalize* the keep-set, not to *build* the mechanism. Build against 435 until it lands.
- **Kyle's primary system metric** — decides whether the physical-slicing investment is worth it. If the digital twin grades on model size, slicing matters. If it grades on something FP16 already covers (and not size), the slicing payoff may not justify the work. FP16 halves size but doesn't speed up CPU inference; class pruning can improve both size and latency.

