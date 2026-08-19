"""
Phase 2 helper: patch the BirdNET v2.4 audio-model SavedModel so that the
RFFT -> Cast(complex64->float32) pattern in the MEL_SPEC1 / MEL_SPEC2 branches
(which takes just the real part of the FFT, not the magnitude) is replaced
with an algebraically equivalent Const(real-DFT cosine matrix) + MatMul.

real(RFFT(x))_k = sum_n x_n * cos(2*pi*k*n/N)   for k = 0 .. N//2

This is exactly linear in x, so it can be expressed as x @ C where
C[n, k] = cos(2*pi*k*n/N), C shape = [N, N//2+1].

This avoids the RFFT op entirely (tf2onnx's RFFT converter only supports the
RFFT -> ComplexAbs pattern, not RFFT -> Cast), while being numerically
identical to the original graph (verified separately after this patch).

Every FunctionDef in the library that contains the RFFT node for a given
branch is patched (there can be more than one copy of the same subgraph
across different concrete functions, e.g. a top-level "wrapped model"
function alongside the one actually invoked by the 'basic' signature's
StatefulPartitionedCall -- and the numeric suffix on function names is
reassigned at load time, so we match by node content, not by a fixed
function name). Variables/ are copied unchanged.
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import tensor_util

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SRC = str(config.SAVEDMODEL)
DST = str(config.SAVEDMODEL_PATCHED)
BRANCHES = {"MEL_SPEC1": 2048, "MEL_SPEC2": 1024}


def make_dft_cos_matrix(n: int) -> np.ndarray:
    k = np.arange(n // 2 + 1)
    idx = np.arange(n)
    return np.cos(2 * np.pi * np.outer(idx, k) / n).astype(np.float32)


def patch_branch(func, branch: str, fft_len: int) -> None:
    node_map = {n.name: n for n in func.node_def}

    rfft_node = node_map[f"model/{branch}/stft/rfft"]
    cast_node = node_map[f"model/{branch}/Cast"]
    mul_input = rfft_node.input[0]  # the windowed real-valued frame, e.g. ".../stft/mul:z:0"

    const_name = f"model/{branch}/dft_cos_matrix"
    assert const_name not in node_map, f"name collision: {const_name}"

    mat = make_dft_cos_matrix(fft_len)
    tensor_proto = tensor_util.make_tensor_proto(mat, dtype=tf.float32)

    const_node = func.node_def.add()
    const_node.name = const_name
    const_node.op = "Const"
    const_node.attr["dtype"].type = tf.float32.as_datatype_enum
    const_node.attr["value"].tensor.CopyFrom(tensor_proto)

    old_output_shapes = None
    if "_output_shapes" in cast_node.attr:
        old_output_shapes = cast_node.attr["_output_shapes"]

    # mul_input is rank 3 ([batch, frames, N]) while the DFT matrix is rank 2
    # ([N, N//2+1]), so this needs broadcasting matmul (BatchMatMulV2), not
    # plain rank-2-only MatMul.
    cast_node.op = "BatchMatMulV2"
    del cast_node.input[:]
    cast_node.input.extend([mul_input, f"{const_name}:output:0"])
    cast_node.ClearField("attr")
    cast_node.attr["T"].type = tf.float32.as_datatype_enum
    cast_node.attr["adj_x"].b = False
    cast_node.attr["adj_y"].b = False
    if old_output_shapes is not None:
        cast_node.attr["_output_shapes"].CopyFrom(old_output_shapes)

    # BatchMatMulV2's output arg is named "output", not "y" (Cast's arg
    # name), so downstream consumers referencing "<cast_name>:y:0" must be
    # repointed.
    old_ref = f"{cast_node.name}:y:0"
    new_ref = f"{cast_node.name}:output:0"
    updated = 0
    for n in func.node_def:
        for i, inp in enumerate(n.input):
            if inp == old_ref:
                n.input[i] = new_ref
                updated += 1
    print(f"[{branch}] fft_len={fft_len} matrix_shape={mat.shape} "
          f"rewired {updated} downstream reference(s) from '{old_ref}' to '{new_ref}'")


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)

    sm = saved_model_pb2.SavedModel()
    with open(os.path.join(SRC, "saved_model.pb"), "rb") as f:
        sm.ParseFromString(f.read())

    assert len(sm.meta_graphs) == 1, f"expected 1 meta graph, got {len(sm.meta_graphs)}"
    graph_def = sm.meta_graphs[0].graph_def

    patched_any = False
    for func in graph_def.library.function:
        node_names = {n.name for n in func.node_def}
        for branch, fft_len in BRANCHES.items():
            if f"model/{branch}/stft/rfft" in node_names:
                print(f"Patching function '{func.signature.name}' for branch {branch}")
                patch_branch(func, branch, fft_len)
                patched_any = True
    assert patched_any, "no functions found containing the target RFFT nodes"

    out_path = os.path.join(DST, "saved_model.pb")
    with open(out_path, "wb") as f:
        f.write(sm.SerializeToString())

    print(f"\nPatched SavedModel written to: {DST}")


if __name__ == "__main__":
    main()
