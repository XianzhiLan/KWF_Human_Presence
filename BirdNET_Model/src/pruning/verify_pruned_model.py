#!/usr/bin/env python3
"""
verify_pruned_model.py — prove the pruned model is behaviourally identical to
the full model on the classes it retains.

This is the gate that matters. Slicing the weight matrix in numpy is easy to get
right; what this checks is that the *whole graph* still behaves, end to end,
through the real preprocessing and the patched spectrogram path.

For every input it asserts:
    pruned_out[:, j]  ==  full_out[:, keep[j]]     exactly, for all j

USAGE
    python verify_pruned_model.py full.onnx pruned.onnx pruned.mapping.csv
    python verify_pruned_model.py full.onnx pruned.onnx pruned.mapping.csv --wav-dir clips/
    python verify_pruned_model.py full.onnx pruned.onnx pruned.mapping.csv --wav-dir clips/ --max-wavs 2000 --shuffle
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import onnxruntime as ort

SAMPLES = 144_000  # 3.000 s @ 48 kHz — the model's fixed input length


def make_session(path):
    so = ort.SessionOptions()
    # single-threaded: near-tie reductions must not vary between the two runs
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def synthetic_batch(rng):
    """Inputs chosen to stress different parts of the graph."""
    cases = {}
    cases["silence"] = np.zeros((1, SAMPLES), np.float32)
    cases["white_noise"] = rng.standard_normal((4, SAMPLES)).astype(np.float32) * 0.1
    cases["full_scale_noise"] = rng.uniform(-1, 1, (2, SAMPLES)).astype(np.float32)

    t = np.arange(SAMPLES, dtype=np.float32) / 48_000.0
    sweep = np.sin(2 * np.pi * (200 + (8000 - 200) * t / 3.0) * t)
    cases["sine_sweep"] = sweep[None, :].astype(np.float32)

    tone = 0.5 * np.sin(2 * np.pi * 4000 * t)
    cases["pure_tone_4k"] = tone[None, :].astype(np.float32)

    cases["dc_offset"] = np.full((1, SAMPLES), 0.5, np.float32)
    cases["tiny_signal"] = (rng.standard_normal((1, SAMPLES)) * 1e-6).astype(np.float32)
    return cases


def load_wavs(wav_dir, limit, shuffle=False, seed=0):
    import soundfile as sf
    files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    total = len(files)
    if shuffle:
        import random
        random.Random(seed).shuffle(files)
    files = files[:limit]
    out = []
    for p in files:
        data, sr = sf.read(p, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if len(data) != SAMPLES:
            print(f"    skipping {os.path.basename(p)}: {len(data)} samples, need {SAMPLES}")
            continue
        out.append((os.path.basename(p), data[None, :]))
    # report spread, so a narrow sample is visible rather than implied
    prefixes = {}
    for nm, _ in out:
        prefixes[nm.rsplit("_", 2)[0]] = prefixes.get(nm.rsplit("_", 2)[0], 0) + 1
    if prefixes:
        spread = ", ".join(f"{k}={v}" for k, v in sorted(prefixes.items()))
        print(f"    sampled from {total} files in dir  |  {spread}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full")
    ap.add_argument("pruned")
    ap.add_argument("mapping")
    ap.add_argument("--wav-dir")
    ap.add_argument("--max-wavs", type=int, default=200)
    ap.add_argument("--shuffle", action="store_true",
                    help="random sample instead of first-N (spreads across recorders/times)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision-mode", action="store_true",
                    help="comparing two different precisions (e.g. FP32 vs FP16): pass on "
                         "argmax agreement rather than bit-identity")
    ap.add_argument("--min-agreement", type=float, default=99.0,
                    help="with --precision-mode, minimum %% top-1 agreement to pass")
    args = ap.parse_args()

    with open(args.mapping, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keep = [int(r["original_en_us_index"]) for r in rows]
    species = [r["species"] for r in rows]
    print(f"mapping: {len(keep)} retained classes\n")

    full, pruned = make_session(args.full), make_session(args.pruned)
    fin, pin = full.get_inputs()[0].name, pruned.get_inputs()[0].name
    print(f"full   : {full.get_inputs()[0].shape} -> {full.get_outputs()[0].shape}")
    print(f"pruned : {pruned.get_inputs()[0].shape} -> {pruned.get_outputs()[0].shape}")

    # If the first model is ALSO already pruned to the same width, the mapping's
    # original-space indices don't apply — the two models share a class space and
    # compare position for position. This is the FP32-pruned vs FP16-pruned case.
    ref_width = full.get_outputs()[0].shape[-1]
    if isinstance(ref_width, int) and ref_width == len(keep):
        keep = list(range(len(keep)))
        print(f"\nboth models have {ref_width} outputs — comparing position for position\n"
              f"(the mapping is used only to name species, not to re-index)")
    print()

    rng = np.random.default_rng(0)
    cases = list(synthetic_batch(rng).items())
    if args.wav_dir:
        wavs = load_wavs(args.wav_dir, args.max_wavs, args.shuffle, args.seed)
        print(f"loaded {len(wavs)} real clips from {args.wav_dir}\n")
        cases += wavs

    print(f"{'case':<20} {'rows':>5} {'max|diff|':>12} {'argmax match':>13}  {'NaN/Inf':>8}")
    print("-" * 64)

    worst = 0.0
    total = agree = 0
    failures = []

    for name, x in cases:
        fo = full.run(None, {fin: x})[0]
        po = pruned.run(None, {pin: x})[0]

        ref = fo[:, keep]
        d = np.abs(ref - po)
        worst = max(worst, float(d.max()))

        # argmax within the retained set must also agree
        a_ref = ref.argmax(axis=1)
        a_got = po.argmax(axis=1)
        m = int((a_ref == a_got).sum())
        agree += m
        total += len(a_ref)

        bad = (not np.isfinite(po).all()) or (not np.isfinite(fo).all())
        status = "yes" if bad else "no"
        print(f"{name:<20} {len(x):>5} {d.max():>12.3e} {m:>6}/{len(a_ref):<6} {status:>8}")
        if bad or (m != len(a_ref) if args.precision_mode else (d.max() != 0.0 or m != len(a_ref))):
            failures.append(name)

    print("-" * 64)
    print(f"\ncases          : {len(cases)}  ({total} rows)")
    print(f"max |diff|     : {worst:.3e}")
    print(f"argmax agree   : {agree}/{total} ({100*agree/total:.3f}%)")

    if args.precision_mode:
        pct = 100 * agree / total
        shown = failures[:15]
        if failures:
            print(f"\ntop-1 flips   : {len(failures)} case(s)")
            for f in shown:
                print(f"    {f}")
            if len(failures) > len(shown):
                print(f"    ... and {len(failures)-len(shown)} more")
        if pct >= args.min_agreement:
            print(f"\nPASS — {pct:.3f}% top-1 agreement (threshold {args.min_agreement}%).")
            print("       Nonzero logit differences are expected between precisions;")
            print("       flips occur where the top-2 margin is below the rounding error.")
        else:
            print(f"\nFAIL — {pct:.3f}% top-1 agreement is below the {args.min_agreement}% threshold.")
            sys.exit(1)
    elif failures:
        shown = failures[:15]
        print(f"\nFAIL — {len(failures)} case(s) differed:")
        for f in shown:
            print(f"    {f}")
        if len(failures) > len(shown):
            print(f"    ... and {len(failures)-len(shown)} more")
        print("\n       Expected exact equality here. If you are comparing two different")
        print("       precisions, re-run with --precision-mode.")
        sys.exit(1)
    else:
        print("\nPASS — pruned model is bit-identical to the full model on all retained classes.")
        print("       Slicing changed the file size and nothing else.")

    # sanity: show what the top class looks like through the new label file
    x = cases[1][1][:1]
    po = pruned.run(None, {pin: x})[0]
    j = int(po[0].argmax())
    print(f"\nspot check: pruned index {j} -> {species[j]!r} "
          f"(original en_us index {keep[j]})")


if __name__ == "__main__":
    main()
