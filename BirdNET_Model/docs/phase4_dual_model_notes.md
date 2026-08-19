# Phase 4 dual-model coverage — assumptions and verifications

Companion to `phase4_dual_model_findings.md` and the machine-readable
`phase4_results_all_recorders.json`, `phase4_negative_check.json`, and per-recorder
`phase4_detail_*.csv`. This note separates what was **measured** from what was
**assumed**, so a reader can tell which conclusions rest on evidence and which rest
on inherited convention.

## Verified in this run (not assumed)

| Property | How it was established |
|---|---|
| Both label files have exactly 493 entries | Counted at startup, fatal assert |
| FP32 and FP16 label files are byte-identical | SHA256 of both, compared; `72b910ab9eae575b…` |
| Neither model is paired with `en_us.txt` / `labels.txt` | Filename check, fatal assert |
| Both models emit 493 classes | Read from the ORT session output shape, fatal assert |
| Allow-list mask is exactly 218 over the 493 space | Built by **name** match against the 493 labels, fatal assert |
| Both models take and return float32 | Read from session I/O types; the identical array is fed to both, no cast |
| The FP16 model really is FP16 | 148 of 148 float initializers are `float16`, 0 remain `float32`. The FP32 model is the mirror image (0 float16, 148 float32) |
| No pre-fix audio in the tree | Stage 1 size census: 0 files at 295,170 or 286,978 bytes |
| No duplicate basenames | Stage 1 halts on any basename resolving to two paths; none found |
| Recorder parsed from filename == CSV `Recorder` column | Checked on all 108,069 CSV rows; 0 disagreements |
| The reference CSV is threshold-filtered at 0.25 | Minimum confidence is exactly 0.250000, on every recorder, with 0 rows below |

That last row matters more than it looks. The 0.25 detection threshold used in
Stages 5 and 6 was specified independently, and it turns out to be exactly the
reference pipeline's own inclusion floor. The negative check is therefore asking a
well-posed question rather than an arbitrary one.

## Assumed, carried over, or unverifiable

1. **288,044 bytes implies 144,000 samples.** Stage 1 buckets by `st_size` only and
   never decodes, so it infers frame count from file size (144,000 × 2 bytes + a
   44-byte canonical WAV header). A file carrying extra metadata chunks that landed
   on exactly this size would be mis-bucketed. This is *checked at scoring time*:
   every clip actually scored is asserted to preprocess to shape `(1, 144000)`, which
   does decode. So the assumption is load-bearing only for the inventory counts, not
   for any scored result.

2. **Reference top-1 is `sp[0]` of the parsed `species` cell.** Multi-species cells
   are resolved by taking the first entry. This matches every prior validated driver
   in the project, but it was not re-derived from the reference generation script,
   which is not in the repo.

3. **`CONF_TOL = 1e-3` defines a confidence match.** Inherited from the validated
   drivers, not derived from any documented tolerance of the reference pipeline. It
   is the reason FP16 conf-match reads low (see the findings); the threshold is
   tighter than FP16's weight-rounding error, by design.

4. **`flat_sigmoid(sensitivity=-1.0)` reproduces the reference postprocessing.**
   Established in earlier phases against `birdnet` v0.1.7 source and reused unchanged.

5. **The FP16 model is the FP16 counterpart of this exact pruned graph.** Supported by
   matching output dimension, byte-identical labels, and a clean float16 initializer
   census — but not proven by a node-level graph diff against the FP32 file.

6. **Stage 6's premise: "absent from the CSV" implies "no detection at or above 0.25."**
   Half of this is now verified (the CSV floor really is 0.25). The other half — that
   the reference pipeline was actually run over every clip sitting on disk — is not
   verifiable from anything available here, and the Audio_Moth_1 result is direct
   evidence that it does not hold universally. See the findings for what that implies.

7. **Threads pinned to 1** (`intra_op = inter_op = 1`) for both sessions. This is a
   deliberate determinism choice, not a performance one: Stage 5 compares two models
   at a hard 0.25 boundary, and nondeterministic reduction order could manufacture a
   set difference that would read as an FP16 effect. Throughput figures in the results
   are therefore single-threaded and are not a latency benchmark.

8. **Stage 6 sampling** uses `random.Random(20260809)`, seeded per recorder, over the
   sorted non-CSV pool. Audio_Moth_1's pool holds only 149 clips, fewer than the 2,000
   target, so it was scored in full rather than sampled.

9. **AM3 and AM5 negative pools are the extraction sample, not the recorder's full
   non-CSV population.** `selective_extract.py` pulled 2,000 non-CSV members per archive
   using the same seed, out of 25,688 (AM3) and 25,246 (AM5) available. Stage 6 then saw
   a pool of exactly 2,000 and sampled all of it. The sampling therefore happened once,
   at extraction, rather than twice — statistically equivalent, but it means those two
   pools cannot be re-sampled without re-extracting.

## Pass B specifics

Pass B (AM3, AM5) ran after the Pass A archives resolved, under the same models,
labels, thresholds, and seed. It was invoked with `--only`, which scores the named
recorders and **carries the other four forward from their existing
`phase4_results_{recorder}.json`** rather than recomputing them. Stage 5 statistics are
merged rather than recomputed: counts are additive, and the maximum-deviation and
closest-to-boundary records are extremal, so the merged values are exact rather than
approximate. Pass A's per-clip detail CSVs were not rewritten.

The two archives are single-date exports covering 2025-03-19 into 03-20. Every member
matching a CSV clip name was extracted, so the shortfall against those recorders' full
CSV row counts is a property of the archives, not of the extraction.
