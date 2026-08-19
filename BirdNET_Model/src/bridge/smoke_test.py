"""End-to-end smoke test using a synthetic BirdNET-shaped model.

Verifies the service wiring without needing the real 18 MB ONNX file: builds a
stand-in model with the same interface contract (input (dynamic, 144000) float32,
output (dynamic, 493) logits), then exercises every endpoint and every startup
gate.

    python smoke_test.py

Passing this proves the plumbing. It does NOT prove numerical parity — that is
what verify_pruned_model.py and the Phase 4 driver are for.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import soundfile as sf
from onnx import TensorProto, helper

FIXTURES = Path(tempfile.mkdtemp(prefix="bridge_fixtures_"))
N_CLASSES = 493
WINDOW = 144_000
SR = 48_000


def build_stub_model(path: Path, n_classes: int = N_CLASSES) -> None:
    """A model with BirdNET's I/O contract and none of its weights."""
    inp = helper.make_tensor_value_info("inputs", TensorProto.FLOAT, [None, WINDOW])
    out = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, n_classes])

    rng = np.random.default_rng(0)
    w = helper.make_tensor("W", TensorProto.FLOAT, [1, n_classes],
                           rng.normal(0, 2, size=(1, n_classes)).astype(np.float32).ravel())

    nodes = [
        helper.make_node("ReduceMean", ["inputs"], ["pooled"], axes=[1], keepdims=1),
        helper.make_node("MatMul", ["pooled", "W"], ["scores"]),
    ]
    graph = helper.make_graph(nodes, "birdnet_stub", [inp], [out], initializer=[w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def build_fixtures() -> dict:
    models_dir = FIXTURES / "models"
    clips_dir = FIXTURES / "clips"
    models_dir.mkdir(parents=True)
    clips_dir.mkdir(parents=True)

    stem = "birdnet_fp16_pruned493"
    model_path = models_dir / f"{stem}.onnx"
    labels_path = models_dir / f"{stem}.labels.txt"
    allow_path = models_dir / "geomodel_allowlist_218.txt"

    build_stub_model(model_path)
    labels = [f"Genus{i:04d} species_Common Bird {i}" for i in range(N_CLASSES)]
    labels_path.write_text("\n".join(labels), encoding="utf-8")
    allow_path.write_text("\n".join(labels[:218]), encoding="utf-8")

    # A model whose class count disagrees with the labels file, for gate 2.
    # Deliberately named WITHOUT "pruned" so it reaches gate 2 rather than being
    # caught earlier by the sidecar-naming gate.
    bad_model = models_dir / "birdnet_stub400.onnx"
    build_stub_model(bad_model, n_classes=400)

    rng = np.random.default_rng(1)
    for i in range(6):
        sf.write(clips_dir / f"clip_{i:03d}.wav",
                 rng.normal(0, 0.05, WINDOW).astype(np.float32), SR, subtype="FLOAT")
    # One deliberately wrong-length clip: the old millisecond-rounding signature.
    sf.write(clips_dir / "WRONG_LENGTH.wav",
             rng.normal(0, 0.05, 147_456).astype(np.float32), SR, subtype="FLOAT")

    config = FIXTURES / "models.json"
    config.write_text(json.dumps({
        "variants": {
            "fp16_pruned493": {
                "model": str(model_path),
                "labels": str(labels_path),
                "allow_list": str(allow_path),
            }
        }
    }), encoding="utf-8")

    return {"config": config, "clips": clips_dir, "models_dir": models_dir,
            "labels": labels_path, "bad_model": bad_model}


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
    return condition


def main() -> int:
    fx = build_fixtures()
    print(f"fixtures: {FIXTURES}\n")
    ok = True

    # --- startup gates ------------------------------------------------------
    print("Startup gates")
    from app.registry import Registry, RegistryError, load_variant

    try:
        load_variant("bad", {"model": str(fx["bad_model"]), "labels": str(fx["labels"])})
        ok &= check("label/output mismatch rejected", False, "no error raised")
    except RegistryError as exc:
        ok &= check("label/output mismatch rejected", "not a matched pair" in str(exc))

    banned = fx["models_dir"] / "labels.txt"
    banned.write_text("\n".join(f"x{i}" for i in range(6362)), encoding="utf-8")
    try:
        load_variant("banned", {"model": str(fx["bad_model"]), "labels": str(banned)})
        ok &= check("deprecated labels.txt rejected", False, "no error raised")
    except RegistryError as exc:
        ok &= check("deprecated labels.txt rejected", "not valid for" in str(exc))

    reg = Registry()
    reg.load_from_config(fx["config"])
    m = reg.get("fp16_pruned493")
    ok &= check("valid pair loads", m.num_classes == N_CLASSES, f"{m.num_classes} classes")
    ok &= check("allow-list containment", m.allow_list_size() == 218,
                f"{m.allow_list_size()} species")

    # --- audio gate ---------------------------------------------------------
    print("\nAudio gates")
    from app.audio import AudioError, read_windows
    try:
        read_windows(fx["clips"] / "WRONG_LENGTH.wav")
        ok &= check("wrong-length clip rejected", False, "no error raised")
    except AudioError as exc:
        ok &= check("wrong-length clip rejected", "not a multiple" in str(exc))

    wins = read_windows(fx["clips"] / "clip_000.wav")
    ok &= check("correct-length clip yields one window",
                len(wins) == 1 and wins[0][0].shape == (WINDOW,))

    # --- API ----------------------------------------------------------------
    print("\nAPI")
    import os
    os.environ["BRIDGE_MODELS_CONFIG"] = str(fx["config"])
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        h = client.get("/health").json()
        ok &= check("GET /health", h["ready"] and "fp16_pruned493" in h["variants_loaded"])

        mods = client.get("/models").json()
        ok &= check("GET /models", mods[0]["num_classes"] == N_CLASSES,
                    f"{mods[0]['file_mb']} MB")

        # 7 files on disk: 6 valid + WRONG_LENGTH.wav (which sorts first).
        r = client.post("/jobs", json={
            "paths": [str(fx["clips"])],
            "variant": "fp16_pruned493",
            "min_confidence": 0.25,
        })
        ok &= check("POST /jobs accepted", r.status_code == 202)
        job_id = r.json()["job_id"]

        state = {}
        for _ in range(100):
            state = client.get(f"/jobs/{job_id}").json()
            if state["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.1)

        ok &= check("job completed", state.get("status") == "done", state.get("status"))
        ok &= check("all 7 files attempted", state.get("completed") == 7)

        met = state.get("metrics") or {}
        ok &= check("bad clip excluded from metrics, run survived",
                    met.get("windows") == 6 and met.get("inference_ms_p95") is not None,
                    f"p50={met.get('inference_ms_p50')}ms "
                    f"p95={met.get('inference_ms_p95')}ms "
                    f"rss={met.get('peak_rss_mb')}MB")

        dets = state.get("detections") or []
        ok &= check("detections carry allow-list flag",
                    all("in_allow_list" in d for d in dets), f"{len(dets)} detections")

        ok &= check("GET /jobs/{unknown} -> 404",
                    client.get("/jobs/nope").status_code == 404)
        ok &= check("DELETE finished job -> 409",
                    client.delete(f"/jobs/{job_id}").status_code == 409)

        with client.stream("GET", f"/jobs/{job_id}/stream") as s:
            first = next(s.iter_lines())
            ok &= check("SSE stream emits frames", first.startswith("event:"), first)

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
