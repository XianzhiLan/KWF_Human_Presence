"""Stage 2 parity check — does the live bridge reproduce the validated driver?

Standalone script, not a pytest suite. Runs against an ALREADY-RUNNING service
(http://127.0.0.1:8000 by default) using real HTTP calls. Does not start, stop,
restart, or reconfigure it, and never touches models.json.

Ground truth: verification_data/phase4_detail_Audio_Moth_6.csv, 709 clips,
produced by src/validation/phase4_dual_model.py -- the driver that reproduced
the birdnet v0.1.7 reference CSV at 100.000% top-1 and confidence agreement.
For each clip and each of {fp32, fp16}, that CSV records exactly one
authoritative (species, probability) pair: the allow-list-restricted argmax
(`{tag}_allow_top1`, `{tag}_allow_conf`). It does not record a full
493-vector, so that single pair per clip per model is the entire comparison
surface available -- not a shortcut, the ceiling of what this CSV can prove.

Threshold note: the CSV's target species is virtually always the same species
the reference itself reported, and the reference CSV's own minimum confidence
across all 108,069 rows is exactly 0.250000. FP32 allow-top1 agrees with the
reference on 100% of AM6 clips, so its target confidence is >= 0.25 essentially
by construction. FP16 agrees on ~99.85%, so a small number of rows could in
principle have a target confidence below 0.25 while still being the local
argmax winner. The bridge only ever returns thresholded detections (there is no
"give me this species' raw score" endpoint), and its lowest reachable threshold
via the public API is INTERNAL_FLOOR=0.1 (min_confidence is clamped up to it
server-side). So this script requests with min_confidence=0.0, which resolves
server-side to an effective 0.1, to maximize the chance every CSV target species
is actually returned. Anything still missing is reported explicitly, along with
its CSV confidence, so a below-0.1 target (structurally unobservable via this
API) is distinguishable from a real bridge/driver disagreement.

Comparison is bit-exact: both sides are full-precision float64 (Python's
`float()` on the CSV string, JSON-decoded float from the bridge's response),
joined with no rounding on either side.

Pass condition: max absolute difference <= 1e-6, per variant. Not widened.

This repo does not include audio -- field recordings are not published. Point
this script at your own copy of the Audio_Moth_6 3-second/48kHz clips with
--audio-root or BIRDNET_AUDIO_ROOT; it fails with instructions, not a bare
FileNotFoundError, if neither is set.

Usage:
    python verify_bridge_parity.py --audio-root /path/to/Audio_Moth_6/clips
    python verify_bridge_parity.py --variant fp32_pruned493 --audio-root ...
    python verify_bridge_parity.py --host http://127.0.0.1:8000 --audio-root ...

    # or via environment variable instead of --audio-root:
    BIRDNET_AUDIO_ROOT=/path/to/clips python verify_bridge_parity.py
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

DETAIL_CSV = Path(os.environ.get(
    "BIRDNET_DETAIL_CSV",
    Path(__file__).resolve().parent.parent.parent / "verification_data" / "phase4_detail_Audio_Moth_6.csv",
))

# variant name (as loaded by the bridge) -> column prefix in the detail CSV
VARIANT_TAG = {
    "fp32_pruned493": "fp32",
    "fp16_pruned493": "fp16",
}

MIN_CONFIDENCE = 0.0     # server clamps to INTERNAL_FLOOR=0.1 -- see module docstring
JOB_POLL_S = 0.5
JOB_TIMEOUT_S = 600.0
PASS_TOL = 1e-6


def resolve_audio_root(cli_value: Path | None) -> Path:
    """Fail loudly and specifically, not with a bare FileNotFoundError on the
    first clip. Anyone cloning this repo has no audio at all -- it's never
    published -- so this is the expected first thing they hit, and it should
    read as instructions, not a crash.
    """
    root = cli_value or os.environ.get("BIRDNET_AUDIO_ROOT")
    if not root:
        raise SystemExit(
            "No audio root configured, and none is bundled with this repo "
            "(field recordings are not published). Point this script at your "
            "own copy of the Audio_Moth_6 3s/48kHz clips with either:\n"
            "  --audio-root <path>\n"
            "  or the BIRDNET_AUDIO_ROOT environment variable"
        )
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(
            f"--audio-root/BIRDNET_AUDIO_ROOT points at '{root}', which is not "
            f"a directory. Set --audio-root <path> or BIRDNET_AUDIO_ROOT to the "
            f"folder containing the Audio_Moth_6 clips."
        )
    return root


def load_ground_truth(tag: str) -> dict[str, tuple[str, float]]:
    """clip -> (species, confidence) for one model tag, from the detail CSV."""
    out: dict[str, tuple[str, float]] = {}
    with open(DETAIL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            species = row[f"{tag}_allow_top1"]
            conf = float(row[f"{tag}_allow_conf"])
            out[row["clip"]] = (species, conf)
    return out


def resolve_paths(clip_names: list[str], audio_root: Path) -> list[str]:
    paths = []
    missing = []
    for name in clip_names:
        p = audio_root / name
        if p.is_file():
            paths.append(str(p))
        else:
            missing.append(name)
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} of {len(clip_names)} clips from {DETAIL_CSV.name} "
            f"are not on disk under {audio_root}. First few: {missing[:5]}"
        )
    return paths


def run_job(client: httpx.Client, variant: str, paths: list[str]) -> dict:
    r = client.post("/jobs", json={
        "paths": paths,
        "variant": variant,
        "min_confidence": MIN_CONFIDENCE,
        "apply_allow_list": True,
        "collect_detections": True,
    })
    r.raise_for_status()
    job_id = r.json()["job_id"]

    t0 = time.perf_counter()
    while True:
        state = client.get(f"/jobs/{job_id}").json()
        if state["status"] in ("done", "error", "cancelled"):
            break
        if time.perf_counter() - t0 > JOB_TIMEOUT_S:
            raise SystemExit(f"FATAL: job {job_id} did not finish within {JOB_TIMEOUT_S}s.")
        time.sleep(JOB_POLL_S)

    if state["status"] != "done":
        raise SystemExit(f"FATAL: job {job_id} ended as {state['status']}: {state.get('error')}")
    if state["completed"] != len(paths):
        raise SystemExit(
            f"FATAL: {state['completed']}/{len(paths)} files attempted -- "
            f"some AM6 clips were skipped (see failures on the /jobs/{job_id} response). "
            f"This is itself a parity failure, not scored around."
        )
    return state


def check_variant(client: httpx.Client, variant: str, clip_names: list[str],
                   paths_by_name: dict[str, str]) -> bool:
    tag = VARIANT_TAG[variant]
    truth = load_ground_truth(tag)

    print(f"\n{'=' * 90}\nVARIANT: {variant}  (CSV column prefix: {tag}_)\n{'=' * 90}")

    mods = client.get("/models").json()
    info = next((m for m in mods if m["variant"] == variant), None)
    if info is None:
        raise SystemExit(f"FATAL: {variant} is not loaded. Loaded: {[m['variant'] for m in mods]}")
    if info["num_classes"] != 493:
        raise SystemExit(f"FATAL: {variant} reports {info['num_classes']} classes, expected 493.")
    print(f"  /models: {variant} — {info['num_classes']} classes, {info['file_mb']} MB")

    paths = [paths_by_name[nm] for nm in clip_names]
    state = run_job(client, variant, paths)

    metrics = state.get("metrics") or {}
    intra = metrics.get("intra_op_num_threads")
    inter = metrics.get("inter_op_num_threads")
    print(f"\n  intra_op_num_threads: {intra}")
    print(f"  inter_op_num_threads: {inter}")
    if intra != 1 or inter != 1:
        raise SystemExit(
            f"STOP: {variant} reports intra_op_num_threads={intra}, "
            f"inter_op_num_threads={inter} -- thread pinning was not picked up by this "
            f"instance. Not interpreting any further numbers until this is resolved."
        )

    dets = state.get("detections") or []

    # Group once: clip -> [(species, confidence), ...]. The target species is
    # always an allow-list member by construction, so in_allow_list is not
    # used as a filter here (and if models.json loads no allow_list for this
    # variant, it wouldn't be meaningful anyway).
    by_clip: dict[str, list[tuple[str, float]]] = {}
    for d in dets:
        by_clip.setdefault(d["data_id"], []).append((d["species"], d["confidence"]))

    diffs: list[tuple[str, str, float, float, float]] = []   # clip, species, csv, bridge, |delta|
    missing: list[tuple[str, str, float]] = []                # clip, species, csv_conf
    top1_flips = 0

    for clip in clip_names:
        species, csv_conf = truth[clip]
        clip_dets = by_clip.get(clip, [])
        bridge_conf = next((c for s, c in clip_dets if s == species), None)
        if bridge_conf is None:
            missing.append((clip, species, csv_conf))
            continue
        delta = abs(bridge_conf - csv_conf)
        diffs.append((clip, species, csv_conf, bridge_conf, delta))
        # Did the bridge's own best-scoring species for this clip differ from the CSV's?
        bridge_top1 = max(clip_dets, key=lambda sc: sc[1])[0]
        if bridge_top1 != species:
            top1_flips += 1

    n = len(clip_names)
    print(f"\n  Clips compared:                 {n}")
    print(f"  Target species missing entirely: {len(missing)}")
    if missing:
        below_floor = sum(1 for _, _, c in missing if c < 0.1)
        print(f"    of which CSV confidence < 0.1 (structurally unreachable via the API): {below_floor}")
        print(f"    of which CSV confidence >= 0.1 (real disagreement, not a threshold artifact): "
              f"{len(missing) - below_floor}")
        print("    First 10 missing:")
        for clip, species, c in missing[:10]:
            print(f"      {clip}  {species}  csv_conf={c:.6f}")

    if diffs:
        deltas = sorted((d[4] for d in diffs), reverse=True)
        mean_d = statistics.fmean(deltas)
        p99_d = deltas[max(0, int(len(deltas) * 0.01) - 1)] if len(deltas) > 1 else deltas[0]
        max_d = deltas[0]
        print(f"\n  Max  |csv - bridge|:  {max_d:.3e}")
        print(f"  Mean |csv - bridge|:  {mean_d:.3e}")
        print(f"  P99  |csv - bridge|:  {p99_d:.3e}")

        worst = sorted(diffs, key=lambda d: -d[4])[:10]
        print("\n  10 largest differences:")
        print(f"    {'clip':<38}{'species':<45}{'csv':>12}{'bridge':>12}{'|delta|':>12}")
        for clip, species, c, b, d in worst:
            print(f"    {clip:<38}{species[:43]:<45}{c:>12.6f}{b:>12.6f}{d:>12.3e}")

        print(f"\n  Clips where bridge top-1 species != CSV species: {top1_flips}")

        passed = max_d <= PASS_TOL and not missing
        print(f"\n  PASS (max <= {PASS_TOL:.0e}, zero missing): {passed}")
    else:
        passed = False
        print("\n  No comparisons produced -- every target species was missing. FAIL.")

    return passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--variant", choices=sorted(VARIANT_TAG), default=None,
                     help="Default: run both variants.")
    ap.add_argument("--audio-root", type=Path, default=None,
                     help="Root directory containing the Audio_Moth_6 clips "
                          "(overrides BIRDNET_AUDIO_ROOT). Required -- audio "
                          "is not published in this repo.")
    args = ap.parse_args()
    variants = [args.variant] if args.variant else sorted(VARIANT_TAG)

    audio_root = resolve_audio_root(args.audio_root)

    with open(DETAIL_CSV, encoding="utf-8") as f:
        clip_names = [row["clip"] for row in csv.DictReader(f)]
    print(f"Loaded {len(clip_names)} clip names from {DETAIL_CSV.name}")

    resolved = resolve_paths(clip_names, audio_root)
    paths_by_name = {Path(p).name: p for p in resolved}
    print(f"All {len(resolved)} clips found under {audio_root}")

    all_pass = True
    with httpx.Client(base_url=args.host, timeout=30.0) as client:
        health = client.get("/health").json()
        if not health["ready"]:
            raise SystemExit(f"FATAL: bridge not ready: {health}")
        print(f"\n/health: onnxruntime {health['onnxruntime_version']} | "
              f"providers {health['providers']} | variants loaded: {health['variants_loaded']}")

        for variant in variants:
            if variant not in health["variants_loaded"]:
                print(f"\nSKIP {variant}: not loaded on this instance.")
                all_pass = False
                continue
            ok = check_variant(client, variant, clip_names, paths_by_name)
            all_pass &= ok

    print(f"\n{'=' * 90}\nOVERALL: {'PASS' if all_pass else 'FAIL'}\n{'=' * 90}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
