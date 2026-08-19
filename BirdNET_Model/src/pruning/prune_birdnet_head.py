#!/usr/bin/env python3
"""
prune_birdnet_head.py — class pruning for the BirdNET FP32 ONNX export.

Physically slices the classifier head (CLASS_DENSE_LAYER) down to a keep-set of
species, producing a smaller model plus the label file that MUST travel with it.

Why this is lossless: the head is a bare affine map
    GlobalAveragePool -> Squeeze -> MatMul[1024,6522] -> Add(bias[6522]) -> scores
with NO Softmax anywhere in the graph. Each class score depends only on its own
column of W and its own bias element, so removing columns cannot perturb the
retained ones. The script asserts this (no-softmax check) before touching anything.

USAGE
    # from the resolved scaffold CSV (uses the en_us_index column)
    python prune_birdnet_head.py birdnet_fp32.onnx \
        --keep-csv pruning_scaffold_union.csv \
        --labels en_us.txt \
        -o birdnet_fp32_pruned.onnx

    # or from a plain list of species labels, resolved against en_us.txt
    python prune_birdnet_head.py birdnet_fp32.onnx \
        --keep-labels reference_csv_species_allowlist.txt \
        --labels en_us.txt \
        -o birdnet_fp32_pruned218.onnx

OUTPUTS
    <out>.onnx         the pruned model
    <out>.labels.txt   new_index -> species, in pruned output order
    <out>.mapping.csv  new_index, original_en_us_index, species  (audit trail)

The label file is not optional. A pruned model without its matching label file
is the exact silent-failure mode this whole effort exists to prevent.
"""

import argparse
import csv
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

HEAD_W = "StatefulPartitionedCall/model/CLASS_DENSE_LAYER/MatMul/ReadVariableOp:0"
HEAD_B = "StatefulPartitionedCall/model/CLASS_DENSE_LAYER/BiasAdd/ReadVariableOp:0"


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    lines = [l for l in lines if l.strip()]
    if len(set(lines)) != len(lines):
        die(f"{path} contains duplicate labels; index mapping would be ambiguous.")
    return lines


def resolve_keep(args, labels):
    """Return a sorted, de-duplicated list of original class indices."""
    if args.keep_csv:
        with open(args.keep_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        col = args.index_column
        if not rows or col not in rows[0]:
            die(f"{args.keep_csv} has no '{col}' column.")
        idx, skipped = [], 0
        for r in rows:
            v = (r.get(col) or "").strip()
            if not v:
                skipped += 1
                continue
            idx.append(int(v))
        if skipped:
            print(f"  note: {skipped} row(s) had no {col} and were skipped "
                  f"(expected — these are the unresolved species held for manual review)")
        return sorted(set(idx)), skipped

    # keep-labels path: resolve species strings against the label file
    wanted = [l.strip() for l in open(args.keep_labels, encoding="utf-8").read().splitlines() if l.strip()]
    pos = {lab: i for i, lab in enumerate(labels)}
    # also allow matching on scientific name alone
    sci = {}
    for i, lab in enumerate(labels):
        sci.setdefault(lab.split("_", 1)[0].strip().lower(), i)

    idx, unresolved = [], []
    for w in wanted:
        if w in pos:
            idx.append(pos[w])
        elif w.split("_", 1)[0].strip().lower() in sci:
            idx.append(sci[w.split("_", 1)[0].strip().lower()])
        else:
            unresolved.append(w)
    if unresolved:
        print(f"  WARNING: {len(unresolved)} label(s) did not resolve and were NOT included:")
        for u in unresolved[:20]:
            print(f"     {u}")
        if len(unresolved) > 20:
            print(f"     ... and {len(unresolved)-20} more")
        if not args.allow_unresolved:
            die("refusing to prune with unresolved labels. Fix them, or pass --allow-unresolved "
                "if you have consciously decided to drop them.")
    return sorted(set(idx)), len(unresolved)


def main():
    ap = argparse.ArgumentParser(description="Slice the BirdNET ONNX classifier head to a keep-set.")
    ap.add_argument("model")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--labels", required=True, help="en_us.txt (6,522 lines). NEVER labels.txt.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--keep-csv", help="CSV containing an index column (default: en_us_index)")
    src.add_argument("--keep-labels", help="text file, one species label per line")
    ap.add_argument("--index-column", default="en_us_index")
    ap.add_argument("--allow-unresolved", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("BirdNET class pruning")
    print("=" * 70)

    labels = load_labels(args.labels)
    print(f"  label file    : {args.labels}  ({len(labels)} classes)")

    model = onnx.load(args.model)
    g = model.graph
    disk_before = os.path.getsize(args.model)
    print(f"  model         : {args.model}  ({disk_before/1024/1024:.2f} MB)")

    # ---- GATE: the property that makes slicing lossless -------------------
    softmax = [n.name for n in g.node if "Softmax" in n.op_type]
    if softmax:
        die("graph contains Softmax/LogSoftmax — outputs are cross-normalized, "
            "so slicing WOULD change retained scores. Do not prune this model.")
    print("  gate          : no softmax in graph — outputs are per-class independent  OK")

    inits = {i.name: i for i in g.initializer}
    if HEAD_W not in inits or HEAD_B not in inits:
        die("classifier head tensors not found under expected names. "
            "This script targets the BirdNET v2.4 SavedModel->ONNX export.")

    W = numpy_helper.to_array(inits[HEAD_W])
    b = numpy_helper.to_array(inits[HEAD_B])
    n_classes = W.shape[1]
    print(f"  head          : W{W.shape}  b{b.shape}")

    # ---- GATE: label file must match the model's actual output width ------
    if len(labels) != n_classes:
        die(f"label file has {len(labels)} entries but the model outputs {n_classes} classes.\n"
            f"       This is the labels.txt/en_us.txt mismatch. Use en_us.txt ({n_classes} lines).")
    print(f"  gate          : label count == model output width ({n_classes})  OK")

    # ---- GATE: head weights must not be shared ---------------------------
    for nm in (HEAD_W, HEAD_B):
        users = [n for n in g.node if nm in n.input]
        if len(users) != 1:
            die(f"{nm} is consumed by {len(users)} nodes; slicing could corrupt another path.")
    print("  gate          : head weights used by exactly one node each  OK")

    keep, _ = resolve_keep(args, labels)
    if not keep:
        die("keep-set is empty.")
    if keep[0] < 0 or keep[-1] >= n_classes:
        die(f"keep-set contains an index outside [0, {n_classes}).")
    print(f"  keep-set      : {len(keep)} classes "
          f"(min idx {keep[0]}, max idx {keep[-1]})")

    # ---- slice ------------------------------------------------------------
    W_new = np.ascontiguousarray(W[:, keep])
    b_new = np.ascontiguousarray(b[keep])

    # numeric proof on the weights themselves, before writing anything
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((64, W.shape[0])).astype(np.float32)
    ref = (emb @ W + b)[:, keep]
    got = emb @ W_new + b_new
    if not np.array_equal(ref, got):
        die(f"slice is not bit-identical (max diff {np.abs(ref-got).max():.3e}). Aborting.")
    print("  slice check   : bit-identical on retained classes (max diff 0.000e+00)  OK")

    inits[HEAD_W].CopyFrom(numpy_helper.from_array(W_new, HEAD_W))
    inits[HEAD_B].CopyFrom(numpy_helper.from_array(b_new, HEAD_B))

    # output shape must be updated or downstream consumers read a stale width
    out = g.output[0]
    dims = out.type.tensor_type.shape.dim
    dims[-1].ClearField("dim_param")
    dims[-1].dim_value = len(keep)
    print(f"  output shape  : {out.name} -> [dynamic, {len(keep)}]")

    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    disk_after = os.path.getsize(args.out)

    # ---- the label file that must travel with the model -------------------
    base = os.path.splitext(args.out)[0]
    lab_path, map_path = base + ".labels.txt", base + ".mapping.csv"
    with open(lab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels[i] for i in keep) + "\n")
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["new_index", "original_en_us_index", "species"])
        for new_i, orig_i in enumerate(keep):
            w.writerow([new_i, orig_i, labels[orig_i]])

    print()
    print(f"  wrote {args.out}  ({disk_after/1024/1024:.2f} MB, "
          f"{100*(1-disk_after/disk_before):.1f}% smaller)")
    print(f"  wrote {lab_path}  ({len(keep)} labels, in pruned output order)")
    print(f"  wrote {map_path}")
    print()
    print("  The pruned model's outputs are ONLY interpretable through the new")
    print("  label file. Ship them together; never pair it with en_us.txt.")


if __name__ == "__main__":
    main()
