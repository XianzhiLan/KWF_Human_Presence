#!/usr/bin/env python3
"""
diagnose_fp16_flips.py — decide whether FP32/FP16 top-1 flips actually matter.

A raw top-1 agreement number is misleading on its own. A flip between two classes
that are both far below the detection threshold changes nothing downstream; a flip
between two classes above it changes a reported detection. This separates the two.

It also tests whether pruning tightened the top-2 margins, which would explain a
worse flip rate on a pruned model than on the full one.

USAGE
    python diagnose_fp16_flips.py fp32_pruned.onnx fp16_pruned.onnx mapping.csv \
        --wav-dir "$BIRDNET_AUDIO_ROOT" --max-wavs 2000 --threshold 0.25
"""

import argparse
import csv
import glob
import os

import numpy as np
import onnxruntime as ort

SAMPLES = 144_000


def flat_sigmoid(x, sensitivity=-1.0):
    return 1.0 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))


def sess(p):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fp32")
    ap.add_argument("fp16")
    ap.add_argument("mapping")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--max-wavs", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="production detection threshold (BIRDNET_CONFIDENCE_THRESHOLD)")
    ap.add_argument("--allowlist", help="optional: restrict argmax to these species")
    args = ap.parse_args()

    import soundfile as sf

    rows = list(csv.DictReader(open(args.mapping, encoding="utf-8")))
    species = [r["species"] for r in rows]

    mask = None
    if args.allowlist:
        allow = {l.strip() for l in open(args.allowlist, encoding="utf-8") if l.strip()}
        mask = np.array([s in allow for s in species])
        print(f"allow-list: {mask.sum()} of {len(species)} classes retained for argmax\n")

    a, b = sess(args.fp32), sess(args.fp16)
    na, nb = a.get_inputs()[0].name, b.get_inputs()[0].name

    files = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))[:args.max_wavs]
    print(f"scoring {len(files)} clips through both models "
          f"(progress every 100)...\n")

    import time
    t0 = time.time()
    n = 0
    flips = []
    margins_all, margins_flip = [], []
    above_thresh_32 = above_thresh_16 = 0
    det_changed = []
    set_diffs = []
    n_det32 = n_det16 = 0
    conf_delta_max = 0.0
    closest_to_thresh = float("inf")

    for fi, p in enumerate(files):
        d, sr = sf.read(p, dtype="float32")
        if d.ndim > 1:
            d = d.mean(axis=1)
        if len(d) != SAMPLES:
            continue
        x = d[None, :]
        l32 = a.run(None, {na: x})[0][0]
        l16 = b.run(None, {nb: x})[0][0]
        if mask is not None:
            l32 = np.where(mask, l32, -1e9)
            l16 = np.where(mask, l16, -1e9)

        n += 1
        if n % 100 == 0:
            el = time.time() - t0
            rate = n / el
            eta = (len(files) - fi - 1) / rate if rate else 0
            print(f"   {n}/{len(files)}  ({rate:.1f} clips/s, ~{eta:.0f}s left)  "
                  f"flips so far: {len(flips)}, set-diffs: {len(set_diffs)}")

        # ---- SET-LEVEL: every class above threshold, not just the winner ----
        c32_all = flat_sigmoid(l32)
        c16_all = flat_sigmoid(l16)
        det32 = set(np.flatnonzero(c32_all >= args.threshold).tolist())
        det16 = set(np.flatnonzero(c16_all >= args.threshold).tolist())
        n_det32 += len(det32)
        n_det16 += len(det16)
        conf_delta_max = max(conf_delta_max, float(np.abs(c32_all - c16_all).max()))
        if det32 != det16:
            added = sorted(det16 - det32)
            removed = sorted(det32 - det16)
            set_diffs.append((os.path.basename(p),
                              [(species[i], float(c32_all[i]), float(c16_all[i])) for i in removed],
                              [(species[i], float(c32_all[i]), float(c16_all[i])) for i in added]))
        # closest approach to the threshold boundary, either side
        for cvec in (c32_all, c16_all):
            near = float(np.abs(cvec - args.threshold).min())
            closest_to_thresh = min(closest_to_thresh, near)
        i32, i16 = int(l32.argmax()), int(l16.argmax())
        c32, c16 = flat_sigmoid(l32[i32]), flat_sigmoid(l16[i16])

        top2 = np.partition(l32, -2)[-2:]
        margin = float(top2[1] - top2[0])
        margins_all.append(margin)

        if c32 >= args.threshold:
            above_thresh_32 += 1
        if c16 >= args.threshold:
            above_thresh_16 += 1

        if i32 != i16:
            margins_flip.append(margin)
            matters = (c32 >= args.threshold) or (c16 >= args.threshold)
            flips.append((os.path.basename(p), species[i32], c32, species[i16], c16, margin, matters))
            if matters:
                det_changed.append(flips[-1])

    print(f"clips scored            : {n}")
    print(f"top-1 flips             : {len(flips)}  ({100*len(flips)/max(n,1):.3f}%)")
    print(f"agreement               : {100*(n-len(flips))/max(n,1):.3f}%")
    print()
    print(f"clips with a detection >= {args.threshold}:")
    print(f"   FP32 : {above_thresh_32}   FP16 : {above_thresh_16}")
    print()
    print("=" * 72)
    print(f"FLIPS THAT CHANGE A REPORTED DETECTION (either side >= {args.threshold})")
    print("=" * 72)
    if not det_changed:
        print("   NONE — every flip is between classes below the detection threshold.")
        print("   Operationally these are invisible: no detection is produced either way.")
    else:
        for f in det_changed:
            print(f"   {f[0]}")
            print(f"      FP32 -> {f[1]}  ({f[2]:.4f})")
            print(f"      FP16 -> {f[3]}  ({f[4]:.4f})   margin {f[5]:.5f}")
        print(f"\n   {len(det_changed)} of {len(flips)} flips affect a real detection.")

    print()
    print("=" * 72)
    print(f"SET-LEVEL: ALL classes above {args.threshold}, not just the winner")
    print("=" * 72)
    print(f"   total detections     FP32 : {n_det32}   FP16 : {n_det16}")
    print(f"   clips whose detection SET differs : {len(set_diffs)} of {n} "
          f"({100*len(set_diffs)/max(n,1):.3f}%)")
    print(f"   max |confidence diff| over ALL classes and clips : {conf_delta_max:.3e}")
    print(f"   closest any class came to the {args.threshold} boundary : {closest_to_thresh:.3e}")
    if not set_diffs:
        print()
        print("   IDENTICAL — every clip yields exactly the same set of detected species")
        print("   in both precisions. Not just the top-1: the complete thresholded output.")
    else:
        print()
        for f in set_diffs[:20]:
            print(f"   {f[0]}")
            for s, a_, b_ in f[1]:
                print(f"      LOST in FP16 : {s}  ({a_:.4f} -> {b_:.4f})")
            for s, a_, b_ in f[2]:
                print(f"      GAINED in FP16: {s}  ({a_:.4f} -> {b_:.4f})")
        if len(set_diffs) > 20:
            print(f"   ... and {len(set_diffs)-20} more")

    if margins_all:
        ma = np.array(margins_all)
        print()
        print("=" * 72)
        print("TOP-2 MARGIN DISTRIBUTION (FP32 logits) — is the argmax fragile?")
        print("=" * 72)
        for q in (1, 5, 10, 25, 50):
            print(f"   p{q:<3} {np.percentile(ma, q):.5f}")
        print(f"   mean {ma.mean():.5f}")
        if margins_flip:
            mf = np.array(margins_flip)
            print(f"\n   margin on flipped clips : mean {mf.mean():.5f}, max {mf.max():.5f}")
            print(f"   margin on all clips     : mean {ma.mean():.5f}")
            print(f"\n   {100*(ma < 0.05).mean():.1f}% of clips have a top-2 margin below 0.05,")
            print("   which is the scale of FP16 rounding error on these logits.")

    print()
    print("=" * 72)
    print("BOTTOM LINE")
    print("=" * 72)
    if not det_changed and not set_diffs:
        print(f"   FP16 changes NO pipeline output at threshold {args.threshold}.")
        print("   Top-1 disagreements exist but occur only between classes the pipeline")
        print("   never reports, and the full thresholded detection set is identical.")
    else:
        print(f"   {len(det_changed)} top-1 flip(s) and {len(set_diffs)} set difference(s)")
        print("   involve classes above threshold. These ARE real behaviour changes.")


if __name__ == "__main__":
    main()
