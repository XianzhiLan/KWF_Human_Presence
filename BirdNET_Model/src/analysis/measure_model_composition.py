"""
BirdNET ONNX — model composition measurement and pruning-safety gates.

Regenerates every number cited in birdnet_class_pruning_plan.md's sizing section,
plus the two structural gates that make class pruning safe:

  GATE 1  no Softmax/LogSoftmax anywhere in the graph (per-class independent outputs)
  GATE 2  physical head slicing is bit-identical for retained classes

Usage:
    python measure_model_composition.py birdnet_fp32.onnx [pruning_scaffold_union.csv]

Requires: onnx, numpy.  Read-only inspection; does not modify the model.
Note: any recent `onnx` works for reading. The project's `onnx==1.18.0` pin exists
because of TF 2.15.1's ml_dtypes cap during *conversion*, which does not apply here.
"""

import sys
import csv
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper

HEAD_W = "StatefulPartitionedCall/model/CLASS_DENSE_LAYER/MatMul/ReadVariableOp:0"
HEAD_B = "StatefulPartitionedCall/model/CLASS_DENSE_LAYER/BiasAdd/ReadVariableOp:0"


def categorize(name):
    if "CLASS_DENSE_LAYER" in name:
        return "Classifier head (CLASS_DENSE_LAYER)"
    if "dft_cos_matrix" in name:
        return "DFT cosine matrices (RFFT patch)"
    if "MEL_SPEC" in name or "input/_1" in name:
        return "Mel/spectrogram frontend"
    if "POST_CONV" in name:
        return "POST_CONV (trunk tail)"
    if "BLOCK_" in name:
        return "Conv trunk (BLOCK_*)"
    return "Other"


def main(model_path, scaffold_path=None):
    import os
    disk = os.path.getsize(model_path)
    model = onnx.load(model_path)
    graph = model.graph

    print("=" * 74)
    print("MODEL IDENTITY")
    print("=" * 74)
    print(f"  file            : {model_path}")
    print(f"  size on disk    : {disk:,} bytes ({disk/1024/1024:.2f} MB)")
    print(f"  producer        : {model.producer_name} {model.producer_version}")
    print(f"  opset           : {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
    for spec, label in ((graph.input, "input"), (graph.output, "output")):
        for t in spec:
            dims = [d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in t.type.tensor_type.shape.dim]
            dtype = onnx.TensorProto.DataType.Name(t.type.tensor_type.elem_type)
            print(f"  {label:15} : {t.name} {dims} {dtype}")
    print(f"  nodes           : {len(graph.node)}   initializers: {len(graph.initializer)}")

    # ---- composition ----
    cats = defaultdict(lambda: [0, 0])
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        c = categorize(init.name)
        cats[c][0] += arr.nbytes
        cats[c][1] += 1

    print()
    print("=" * 74)
    print("SIZE COMPOSITION")
    print("=" * 74)
    print(f"  {'Component':40} {'MB':>9} {'% file':>9} {'tensors':>8}")
    print("  " + "-" * 70)
    for c, (b, n) in sorted(cats.items(), key=lambda x: -x[1][0]):
        print(f"  {c:40} {b/1024/1024:9.3f} {100*b/disk:8.1f}% {n:8}")
    total = sum(v[0] for v in cats.values())
    print("  " + "-" * 70)
    print(f"  {'TOTAL initializers':40} {total/1024/1024:9.3f} {100*total/disk:8.1f}%")
    print(f"  {'Non-weight overhead':40} {(disk-total)/1024/1024:9.3f}")

    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    # ---- GATE 1: no softmax ----
    print()
    print("=" * 74)
    print("GATE 1 — per-class independent outputs (no cross-class normalization)")
    print("=" * 74)
    softmaxes = [n.name for n in graph.node if "Softmax" in n.op_type]
    print(f"  Softmax/LogSoftmax nodes found: {len(softmaxes)}")
    if softmaxes:
        print("  *** FAIL — cross-class normalization present; slicing WILL change retained scores")
        for s in softmaxes:
            print("     ", s)
    else:
        print("  PASS — no softmax; each output depends only on its own weight column + bias.")

    producer = {o: n for n in graph.node for o in n.output}
    print("  Output path (traced backward from graph output):")
    cur, chain = graph.output[0].name, []
    for _ in range(6):
        n = producer.get(cur)
        if n is None:
            break
        chain.append(n.op_type)
        nxt = [i for i in n.input if i not in inits]
        if not nxt:
            break
        cur = nxt[0]
    print("     ", " <- ".join(chain))

    # ---- DFT analysis ----
    print()
    print("=" * 74)
    print("DFT MATRICES (RFFT patch) — redundancy and regenerability")
    print("=" * 74)
    dft_total = 0
    for name, arr in inits.items():
        if "dft_cos_matrix" not in name:
            continue
        dft_total += arr.nbytes
        N, K = arr.shape
        n_idx = np.arange(N).reshape(-1, 1)
        k_idx = np.arange(K).reshape(1, -1)
        analytic = np.cos(2 * np.pi * n_idx * k_idx / N).astype(np.float32)
        maxdiff = np.abs(arr - analytic).max()
        uniq = len(np.unique(arr))
        fp16_err = np.abs(arr - arr.astype(np.float16).astype(np.float32))
        tag = name.split("/")[-2] if "/" in name else name
        print(f"  {tag}  shape {arr.shape}  {arr.nbytes/1024/1024:.3f} MB")
        print(f"     unique values : {uniq:,} of {arr.size:,}  ({arr.size/uniq:.0f}x redundant)")
        print(f"     vs cos(2*pi*n*k/N) : max diff {maxdiff:.3e}"
              f"  -> {'ANALYTIC (regenerable)' if maxdiff < 1e-4 else 'NOT plain cosine'}")
        print(f"     FP16 round-trip err: max {fp16_err.max():.3e}  mean {fp16_err.mean():.3e}")
    if dft_total:
        print(f"  DFT total: {dft_total/1024/1024:.3f} MB ({100*dft_total/disk:.1f}% of model)")
        print("  NOTE: ONNX opset 17 has a native DFT op; replacing these constants with it")
        print("        would remove this entirely. Requires bit-parity test + ORT kernel check.")

    # ---- GATE 2: slice equivalence ----
    if HEAD_W not in inits:
        print("\n  (head weights not found under expected name; skipping GATE 2)")
        return
    W, b = inits[HEAD_W], inits[HEAD_B]

    keep = None
    if scaffold_path:
        with open(scaffold_path) as f:
            rows = list(csv.DictReader(f))
        # accept either the scaffold CSV (en_us_index) or a pruned model's
        # mapping CSV (original_en_us_index)
        col = None
        for cand in ("en_us_index", "original_en_us_index"):
            if rows and cand in rows[0]:
                col = cand
                break
        if col is None:
            print(f"\n  (no index column found in {scaffold_path}; expected "
                  f"'en_us_index' or 'original_en_us_index')")
        else:
            keep = sorted({int(r[col]) for r in rows if r[col]})
            print(f"\n  keep-set from {os.path.basename(scaffold_path)} "
                  f"column '{col}': {len(keep)} classes")

    # If the indices refer to the ORIGINAL class space but this model has already
    # been pruned, they are out of range here. Slicing is not meaningful twice.
    if keep and keep[-1] >= W.shape[1]:
        print(f"  note: this model already has {W.shape[1]} outputs, but the index list "
              f"goes up to {keep[-1]} —\n        it describes the ORIGINAL class space, so this "
              f"model appears to be already pruned.\n        Running GATE 2 on a random subset "
              f"instead, to still exercise the check.")
        keep = None

    if not keep:
        n = min(474, W.shape[1])
        keep = sorted(np.random.default_rng(0).choice(W.shape[1], n, replace=False).tolist())
        print(f"  (using a random {n}-class keep-set for the gate)")

    print()
    print("=" * 74)
    print("GATE 2 — physical slicing is lossless for retained classes")
    print("=" * 74)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((256, W.shape[0])).astype(np.float32)
    full = emb @ W + b
    sliced = emb @ W[:, keep] + b[keep]
    diff = np.abs(sliced - full[:, keep])
    print(f"  keep-set size : {len(keep)}")
    print(f"  max abs diff  : {diff.max():.3e}")
    print(f"  bit-identical : {np.array_equal(sliced, full[:, keep])}")
    print(f"  {'PASS' if diff.max() == 0 else '*** FAIL'} — "
          f"slicing {'does not perturb' if diff.max() == 0 else 'PERTURBS'} retained classes")

    # ---- projections ----
    print()
    print("=" * 74)
    print("PRUNING SIZE PROJECTIONS (head-slicing only, FP32)")
    print("=" * 74)
    head_now = W.nbytes + b.nbytes
    print(f"  {'keep-set':>10} {'head MB':>10} {'total MB':>10} {'reduction':>11}")
    print("  " + "-" * 45)
    print(f"  {W.shape[1]:>10} {head_now/1024/1024:10.2f} {disk/1024/1024:10.2f} {'—':>11}")
    for K in (len(keep), 218):
        new_head = W.shape[0] * K * 4 + K * 4
        new_total = disk - (head_now - new_head)
        print(f"  {K:>10} {new_head/1024/1024:10.2f} {new_total/1024/1024:10.2f} "
              f"{100*(1-new_total/disk):10.0f}%")
    print("\n  Keep-set size barely affects the result once slicing happens at all —")
    print("  no size pressure to keep the target list tight.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
