# BirdNET Bridge

HTTP service wrapping the pruned BirdNET ONNX models, built to the endpoint shape
of the DeepData Portal AI sidecar.

**Scope:** BirdNET only. This is not chain integration — HP is the user-facing
output, so this service cannot be pointed at Blackbird's pass/fail scenarios as a
validation target. What it is: a **benchmarking service** that reports size,
latency, CPU time, and memory per model variant on the target device, which is
what the open compression-metric question needs in order to close.

## Setup

```bash
cd src/bridge                                        # models.json.example's paths are relative to here
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp models.json.example models.json                   # edit the paths if your layout differs
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000/docs> for an interactive client — no client code
needed to demo it.

No TensorFlow dependency, so this does not need the Python 3.11 pin that
`birdnet==0.1.7` forces. Any 3.10+ works. The parity check below (see
"Verified") was run under Python 3.11 / `onnxruntime==1.27.0` / `numpy==1.26.4`
specifically, matching the environment that produced the Phase 4 reference
results — that combination is confirmed, not merely expected to work.

`--host 0.0.0.0` is what lets the service run on a Pi 4 and be driven from your
laptop. Drop it to bind localhost only.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/jobs` | Submit an analysis job. Returns a job id immediately. |
| `GET` | `/jobs/{id}` | Status, metrics, detections. |
| `GET` | `/jobs/{id}/stream` | Live `progress` / `result` / `done` / `error` events (SSE). |
| `DELETE` | `/jobs/{id}` | Cancel a running job. |
| `GET` | `/health` | Readiness, loaded variants, ORT version and providers. |
| `GET` | `/models` | Per-variant class count, file size, allow-list size. |

The paths and the four event-type names match Daan's sidecar contract, so client
code written against this carries over to the pack framework rather than being
thrown away.

## Startup gates

All four run once at load. A gate failure raises and **the service does not
start** — a mispaired model must be impossible to serve, not merely unlikely.

1. **Banned label file.** `labels.txt`, or any file with 6,362 entries, is refused
   by name and by count.
2. **Label count equals output dimension.** Falls back to a zero-input probe when
   the output dim is dynamic. This is the check that makes silent mis-indexing
   impossible.
3. **Sidecar naming.** A model whose stem contains `pruned` must be paired with
   its own `<stem>.labels.txt`.
4. **Input contract.** `(dynamic, 144000)` float32 or refuse.

Loading an allow-list additionally re-runs the containment check against the
model's keep-set. That assumption has failed twice on this project, so it is
verified every load rather than inherited.

## Behaviour worth knowing

- **Wrong-length clips are rejected, not padded.** Anything that isn't an exact
  multiple of 144,000 samples raises. Padding would produce a plausible score for
  a window the reference pipeline never evaluated. Bad files are skipped, logged,
  counted in `completed`, and excluded from `windows` and the latency stats.
- **Warm-up runs before timing.** ORT's first inference pays one-time allocation
  and kernel-selection cost several times steady-state. Three discarded runs make
  short and long jobs comparable.
- **Timing brackets `session.run` only.** Preprocessing is measured separately.
- **The allow-list marks, it never filters.** Detections carry
  `in_allow_list: true|false`, so one run reproduces both the allow-list-filtered
  numbers (which match the reference CSV) and the unrestricted ones. Collapsing
  the two is what made raw top-1 look like a regression when it was answering a
  different question.
- **Both ORT thread counts default to 1.** Determinism is the out-of-the-box
  behavior, not an opt-in — a bridge that returns different numbers run-to-run
  on identical input is a worse failure than a slow one, and unpinned threading
  is exactly the failure mode that made an earlier pass of this project's own
  results briefly non-reproducible. Raise `BRIDGE_INTRA_OP_THREADS` and/or
  `BRIDGE_INTER_OP_THREADS` above `1` deliberately, for a throughput benchmark
  where run-to-run identical output isn't the point.

## A/B benchmark

```bash
python benchmark.py \
  --clips "/path/to/filtered_clips_1" \
  --variants fp16_pruned493 fp32_pruned493 \
  --limit 2000 --out bench_pi4.json
```

Run it **on the target device**. Laptop numbers do not transfer to a Pi 4, and
mixing the two is worse than having neither.

## Smoke test

```bash
python smoke_test.py
```

Builds a synthetic model with BirdNET's I/O contract and exercises every endpoint
and every gate. Proves the plumbing; proves nothing about numerical parity —
that remains the job of `verify_pruned_model.py` and the Phase 4 driver.

## Verified

- **`smoke_test.py`: 16/16** on Python 3.11 / ORT 1.27.0 — plumbing, all four
  startup gates, both audio gates. This proves nothing about numerical parity;
  that is a separate, narrower claim, made below.
- **Preprocessing + inference parity (`verify_bridge_parity.py`): bit-exact.**
  Checked against **709 Audio_Moth_6 clips**, one (species, probability) pair
  per clip per model — the allow-list-restricted argmax, which is the entire
  comparison surface `phase4_detail_Audio_Moth_6.csv` records and the ceiling
  of what it can prove. This is **not** a check of the full 493-class output
  vector, and it does **not** cover the 51,720-clip validation set the driver
  itself was validated against — the bridge is verified to reproduce the
  driver on this 709-clip subset, not independently validated at that scale.
  - `fp16_pruned493`: 709/709 clips compared, 0 missing, max/mean/p99 absolute
    difference all `0.000e+00`, 0 top-1 flips.
  - `fp32_pruned493`: 709/709 clips compared, 0 missing, max/mean/p99 absolute
    difference all `0.000e+00`, 0 top-1 flips.
  - Run against the live service with `intra_op_num_threads=1` and
    `inter_op_num_threads=1` confirmed via each job's returned metrics.
  - The exact-zero result rests on comparing full-precision float64 reprs on
    both sides with no rounding anywhere in the join, which is what makes
    "bit-exact" a claim someone can check rather than one they have to trust.
  - Because the match is exact, the bridge **inherits** the Phase 4 driver's
    validation against the birdnet v0.1.7 reference on this subset, rather
    than making an independent correctness claim of its own.
  - Reproduce it yourself: `python verify_bridge_parity.py --audio-root /path/to/Audio_Moth_6/clips`
    (audio is not published in this repo; see `verify_bridge_parity.py`'s
    docstring).

## Threshold, resolved

`INTERNAL_FLOOR = 0.1` in `inference.py` was previously flagged as taken from
the project handoff rather than independently re-read `birdnet` 0.1.7 source.
This is now resolved by measurement: the reference CSV's minimum confidence is
exactly 0.250000 across all 108,069 rows, with nothing below it. Either the
0.1 floor does not exist in the reference call chain, or it never reaches the
output — either way, 0.25 is the reference's own inclusion floor, and the 0.1
constant here is a no-op at any threshold >= 0.1, which covers every
documented use of this service. Kept as a named constant rather than removed,
so a benchmark run below 0.1 fails loudly instead of silently returning
results the reference pipeline never produced.
