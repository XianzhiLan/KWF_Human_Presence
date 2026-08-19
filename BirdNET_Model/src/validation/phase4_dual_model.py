"""Phase 4 recorder-coverage extension, FP32 + FP16 pruned-493, one decode per clip.

Stages 2-6 of the dual-model coverage plan. Stage 1 lives in preflight_inventory.py
and must have been run first; every clip path here resolves through the index it
wrote, so this script never globs the audio tree.

Scoring core is lifted from phase4_pruned493.py and parameterized over
(clip list, model list). No existing validated script is modified.

CRITICAL: the pruned models' index 0 is NOT en_us.txt index 0. These models emit
493 values whose meaning is defined ONLY by their own label file. Pairing them
with en_us.txt (6,522) or labels.txt (6,362) yields confident WRONG species with
no error, so the label identity is asserted before any inference runs.

CLI:
  python phase4_dual_model.py --gate      # Stage 2 only
  python phase4_dual_model.py --run       # Stages 3, 4, 5
  python phase4_dual_model.py --negative  # Stage 6
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

import birdnet_preprocessing as bp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.WORK_DIR
CSV_PATH = config.REFERENCE_CSV
ALLOWLIST = config.ALLOWLIST
INVENTORY = config.INVENTORY_JSON

FP32_MODEL = config.PRUNED_FP32_ONNX
FP32_LABELS = config.PRUNED_FP32_LABELS
FP16_MODEL = config.PRUNED_FP16_ONNX
FP16_LABELS = config.PRUNED_FP16_LABELS

MODELS = ("fp32", "fp16")

TARGET_FRAMES = 144_000
EXPECTED_CLASSES = 493
EXPECTED_ALLOW = 218
CONF_TOL = 1e-3
BATCH = 128
GATE_N = 500
DET_THRESHOLD = 0.25   # Stage 5/6 detection threshold. Deliberately NOT the
                       # existing driver's MIN_CONF = 0.1; that value is not reused.
NEG_SAMPLE = 2000
NEG_SEED = 20260809

TS_RE = re.compile(r"Audio_Moth_\d+_(\d{8})_(\d{6})")


# --------------------------------------------------------------------------- #
# Reference CSV
# --------------------------------------------------------------------------- #
def parse_cell(c: str) -> List[str]:
    try:
        v = ast.literal_eval(c)
        return [str(s).strip() for s in v] if isinstance(v, (list, tuple)) else [str(v).strip()]
    except (ValueError, SyntaxError):
        return [c.strip()]


def load_ref() -> Dict[str, dict]:
    ref: Dict[str, dict] = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sp = parse_cell(row["species"])
            try:
                conf = float(row["confidence"])
            except (ValueError, KeyError):
                conf = None
            ref[row["clip_name"]] = {
                # Multi-species cells: reference top-1 is sp[0], matching the
                # behavior of every prior validated driver.
                "top1": sp[0] if sp else None,
                "conf": conf,
                "recorder": row.get("Recorder", ""),
                "datetime": row.get("Datetime", ""),
            }
    return ref


# --------------------------------------------------------------------------- #
# Fatal assertions — run before any inference
# --------------------------------------------------------------------------- #
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def assert_wiring() -> Tuple[np.ndarray, Dict[str, int], np.ndarray, List[str]]:
    print("=" * 100)
    print("PRE-INFERENCE ASSERTIONS")
    print("=" * 100)

    for tag, p in (("FP32", FP32_LABELS), ("FP16", FP16_LABELS)):
        low = p.name.lower()
        if low in ("en_us.txt", "labels.txt"):
            raise SystemExit(
                f"FATAL: {tag} label path is {p.name}. The pruned models must never be "
                f"paired with en_us.txt (6,522) or labels.txt (6,362)."
            )
        print(f"  {tag} label path: {p.name}  (not en_us.txt / labels.txt) OK")

    lab32 = bp.load_species(FP32_LABELS)
    lab16 = bp.load_species(FP16_LABELS)
    for tag, lab in (("FP32", lab32), ("FP16", lab16)):
        if len(lab) != EXPECTED_CLASSES:
            raise SystemExit(
                f"FATAL: {tag} label file has {len(lab)} entries; expected {EXPECTED_CLASSES}."
            )
    print(f"  Both label files: {EXPECTED_CLASSES} entries OK")

    h32, h16 = sha256(FP32_LABELS), sha256(FP16_LABELS)
    if h32 != h16:
        raise SystemExit(
            "FATAL: FP32 and FP16 label files are not byte-identical.\n"
            f"  FP32 {h32}\n  FP16 {h16}\n"
            "A shared index->species mapping is required to compare the two models."
        )
    print(f"  Label files byte-identical  SHA256 {h32[:32]}... OK")

    species_list = lab32
    sp = np.array(species_list, dtype=object)
    sp2idx = {s: i for i, s in enumerate(species_list)}

    allow_names = [s for s in ALLOWLIST.read_text("utf-8").splitlines() if s]
    missing = [s for s in allow_names if s not in sp2idx]
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} allow-list species absent from the 493 label space. "
            f"First few: {missing[:5]}"
        )
    allow_idx = np.array(sorted(sp2idx[s] for s in allow_names))
    if len(allow_idx) != EXPECTED_ALLOW:
        raise SystemExit(
            f"FATAL: allow-list mask has {len(allow_idx)} entries over the 493 space; "
            f"expected exactly {EXPECTED_ALLOW}."
        )
    print(f"  Allow-list mask over 493 space: {len(allow_idx)} == {EXPECTED_ALLOW} OK")
    return sp, sp2idx, allow_idx, species_list


def build_sessions() -> Dict[str, dict]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    out: Dict[str, dict] = {}
    for tag, path in (("fp32", FP32_MODEL), ("fp16", FP16_MODEL)):
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        odim = s.get_outputs()[0].shape[-1]
        if odim != EXPECTED_CLASSES:
            raise SystemExit(
                f"FATAL: {tag} model output dim is {odim}; expected {EXPECTED_CLASSES}."
            )
        out[tag] = {
            "sess": s,
            "in": s.get_inputs()[0].name,
            "out": s.get_outputs()[0].name,
            "path": path,
        }
        print(f"  {tag.upper()} model output dim {odim} == {EXPECTED_CLASSES} OK  ({path.name})")
    print("  Threads pinned: intra_op=1 inter_op=1 (determinism at the 0.25 boundary)")
    print("=" * 100)
    return out


# --------------------------------------------------------------------------- #
# Decode-once dual-model core
# --------------------------------------------------------------------------- #
def iter_scored(
    clips: List[Tuple[str, Path]],
    sessions: Dict[str, dict],
    label: str,
) -> Iterator[Tuple[str, Dict[str, np.ndarray], float]]:
    """Decode each clip once, feed the identical float32 array to both sessions.

    Yields (clip_name, {model_tag: probs(493,)}, rms).
    """
    buf_names: List[str] = []
    buf_arr: List[np.ndarray] = []
    buf_rms: List[float] = []
    n_done = 0
    total = len(clips)
    t0 = time.perf_counter()

    def run_batch() -> List[Tuple[str, Dict[str, np.ndarray], float]]:
        nonlocal n_done
        if not buf_names:
            return []
        batch = np.concatenate(buf_arr, axis=0).astype(np.float32)
        probs: Dict[str, np.ndarray] = {}
        for tag, m in sessions.items():
            logits = m["sess"].run([m["out"]], {m["in"]: batch})[0]
            if logits.shape[1] != EXPECTED_CLASSES:
                raise SystemExit(
                    f"FATAL: {tag} produced {logits.shape[1]} classes mid-run; "
                    f"expected {EXPECTED_CLASSES}."
                )
            probs[tag] = bp.flat_sigmoid(logits, -1.0)
        out = [
            (nm, {t: probs[t][i] for t in probs}, buf_rms[i])
            for i, nm in enumerate(buf_names)
        ]
        prev, n_done = n_done, n_done + len(buf_names)
        if n_done // 5000 != prev // 5000 or n_done == total:
            el = time.perf_counter() - t0
            rate = n_done / el if el else 0
            eta = (total - n_done) / rate if rate else 0
            print(
                f"    [{label}] {n_done}/{total}  {rate:.1f} clip/s  ETA {eta / 60:.1f} min",
                flush=True,
            )
        buf_names.clear()
        buf_arr.clear()
        buf_rms.clear()
        return out

    for nm, path in clips:
        arr = bp.preprocess_file(path)
        if arr.shape != (1, TARGET_FRAMES):
            raise SystemExit(
                f"FATAL: {nm} preprocessed to {arr.shape}, expected (1, {TARGET_FRAMES}). "
                f"Wrong-length audio is a data error, not a model error. Path: {path}"
            )
        buf_names.append(nm)
        buf_arr.append(arr)
        buf_rms.append(float(np.sqrt(np.mean(arr[0] ** 2))))
        if len(buf_names) >= BATCH:
            yield from run_batch()
    yield from run_batch()


class Stage5Acc:
    """Streaming accumulators for the FP16-vs-FP32 comparison (all 493 classes)."""

    def __init__(self) -> None:
        self.n = 0
        self.identical = 0
        self.diffs: List[dict] = []
        self.max_dev = -1.0
        # Species stored by name, not index, so a prior pass's numbers can be
        # merged in without re-deriving the index space.
        self.max_dev_at: Optional[Tuple[str, str]] = None
        self.closest = 1.0
        self.closest_at: Optional[Tuple[str, str, str, float]] = None

    def merge_prior(self, prev: dict) -> None:
        """Fold a previous pass's Stage 5 summary into this accumulator."""
        self.n += prev["clips_compared"]
        self.identical += prev["identical_sets"]
        self.diffs = prev["differences"] + self.diffs
        if prev["max_abs_conf_deviation"] > self.max_dev:
            self.max_dev = prev["max_abs_conf_deviation"]
            a = prev["max_abs_conf_deviation_at"]
            self.max_dev_at = (a["clip"], a["species"]) if a else None
        if prev["closest_to_threshold"] < self.closest:
            self.closest = prev["closest_to_threshold"]
            b = prev["closest_to_threshold_at"]
            self.closest_at = (b["clip"], b["species"], b["model"], b["conf"]) if b else None

    def update(self, nm: str, p32: np.ndarray, p16: np.ndarray, sp: np.ndarray,
               allow_idx: np.ndarray, set32: set, set16: set) -> None:
        self.n += 1
        if set32 == set16:
            self.identical += 1
        else:
            for s in sorted(set32 ^ set16):
                j = int(np.nonzero(sp == s)[0][0])
                c32, c16 = float(p32[j]), float(p16[j])
                self.diffs.append({
                    "clip": nm, "species": s,
                    "fp32_conf": c32, "fp16_conf": c16,
                    "delta_fp16_minus_fp32": c16 - c32,
                    "in_fp32_set": s in set32, "in_fp16_set": s in set16,
                })
        dev = np.abs(p32 - p16)
        j = int(np.argmax(dev))
        if float(dev[j]) > self.max_dev:
            self.max_dev = float(dev[j])
            self.max_dev_at = (nm, str(sp[j]))
        for tag, p in (("fp32", p32), ("fp16", p16)):
            d = np.abs(p - DET_THRESHOLD)
            k = int(np.argmin(d))
            if float(d[k]) < self.closest:
                self.closest = float(d[k])
                self.closest_at = (nm, str(sp[k]), tag, float(p[k]))


def score_recorder(
    clips: List[Tuple[str, Path]],
    ref: Dict[str, dict],
    sessions: Dict[str, dict],
    sp: np.ndarray,
    sp2idx: Dict[str, int],
    allow_idx: np.ndarray,
    label: str,
    acc: Optional[Stage5Acc] = None,
) -> Tuple[List[dict], float]:
    """Score a clip list under both models. Returns (per-clip rows, seconds)."""
    allow_species = sp[allow_idx]
    rows: List[dict] = []
    t0 = time.perf_counter()

    for nm, probs, rms in iter_scored(clips, sessions, label):
        r = ref[nm]
        ri = sp2idx.get(r["top1"])
        row = {
            "clip": nm,
            "recorder": r["recorder"],
            "datetime": r["datetime"],
            "ref_top1": r["top1"],
            "ref_conf": r["conf"],
            "rms": rms,
        }
        det_sets = {}
        for tag in MODELS:
            p = probs[tag]
            raw_i = int(np.argmax(p))
            allow_i = int(allow_idx[int(np.argmax(p[allow_idx]))])
            our_ref_conf = float(p[ri]) if ri is not None else None
            det = set(allow_species[p[allow_idx] >= DET_THRESHOLD].tolist())
            det_sets[tag] = det
            row.update({
                f"{tag}_raw_top1": str(sp[raw_i]),
                f"{tag}_raw_conf": float(p[raw_i]),
                f"{tag}_allow_top1": str(sp[allow_i]),
                f"{tag}_allow_conf": float(p[allow_i]),
                f"{tag}_conf_for_ref": our_ref_conf,
                f"{tag}_raw_match": str(sp[raw_i]) == r["top1"],
                f"{tag}_allow_match": str(sp[allow_i]) == r["top1"],
                f"{tag}_conf_match": (
                    our_ref_conf is not None
                    and r["conf"] is not None
                    and abs(our_ref_conf - r["conf"]) < CONF_TOL
                ),
                f"{tag}_det_set": ";".join(sorted(det)),
                f"{tag}_det_n": len(det),
            })
        row["det_sets_identical"] = det_sets["fp32"] == det_sets["fp16"]
        rows.append(row)
        if acc is not None:
            acc.update(nm, probs["fp32"], probs["fp16"], sp, allow_idx,
                       det_sets["fp32"], det_sets["fp16"])

    return rows, time.perf_counter() - t0


def summarize(rows: List[dict], tag: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"clips": 0}
    allow = sum(r[f"{tag}_allow_match"] for r in rows)
    raw = sum(r[f"{tag}_raw_match"] for r in rows)
    cm = sum(r[f"{tag}_conf_match"] for r in rows)
    return {
        "clips": n,
        "allow_top1_n": allow, "allow_top1_pct": 100 * allow / n,
        "raw_top1_n": raw, "raw_top1_pct": 100 * raw / n,
        "conf_match_n": cm, "conf_match_pct": 100 * cm / n,
    }


def date_coverage(rows: List[dict]) -> str:
    dates = set()
    for r in rows:
        m = TS_RE.search(r["clip"])
        if m:
            d = m.group(1)
            dates.add(f"{d[:4]}-{d[4:6]}-{d[6:]}")
    if not dates:
        return "unknown"
    ds = sorted(dates)
    return ds[0] + " (single date)" if len(ds) == 1 else f"{ds[0]}..{ds[-1]} ({len(ds)} dates)"


# --------------------------------------------------------------------------- #
# Inventory plumbing
# --------------------------------------------------------------------------- #
def load_inventory() -> dict:
    if not INVENTORY.exists():
        raise SystemExit(f"FATAL: {INVENTORY} missing. Run preflight_inventory.py first.")
    with open(INVENTORY, encoding="utf-8") as f:
        return json.load(f)


def clips_for(inv: dict, rec: str, key: str) -> List[Tuple[str, Path]]:
    idx = inv["index"]
    out = []
    for nm in inv["recorders"][rec][key]:
        p = idx.get(nm)
        if p is None:
            raise SystemExit(
                f"FATAL: {nm} is in the Stage 1 {key} list but absent from the path index. "
                f"The inventory is stale — re-run preflight_inventory.py."
            )
        out.append((nm, Path(p)))
    return out


def write_detail(rows: List[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #
def stage2(inv, ref, sessions, sp, sp2idx, allow_idx) -> None:
    order = sorted(inv["passing"], key=lambda r: inv["recorders"][r]["scoreable"])
    if not order:
        raise SystemExit("FATAL: no recorder cleared Stage 1.")
    rec = order[0]
    clips = clips_for(inv, rec, "scoreable_names")[:GATE_N]
    print(f"\n=== STAGE 2 GATE: {len(clips)} clips from {rec} "
          f"(smallest passing, scoreable={inv['recorders'][rec]['scoreable']}) ===")

    rows, el = score_recorder(clips, ref, sessions, sp, sp2idx, allow_idx, f"gate/{rec}")
    print(f"\nGate scored {len(rows)} clips in {el:.1f}s")
    print(f"\n{'model':<8}{'allow-list top-1':>18}{'conf-match':>14}{'raw top-1':>12}")
    s = {}
    for tag in MODELS:
        s[tag] = summarize(rows, tag)
        print(f"{tag.upper():<8}{s[tag]['allow_top1_pct']:>17.3f}%"
              f"{s[tag]['conf_match_pct']:>13.3f}%{s[tag]['raw_top1_pct']:>11.3f}%")

    if s["fp32"]["allow_top1_pct"] != 100.0:
        bad = [r for r in rows if not r["fp32_allow_match"]]
        print(f"\n*** GATE FAILED: FP32 allow-list top-1 = "
              f"{s['fp32']['allow_top1_pct']:.3f}%, must be 100.000% ***")
        print(f"{len(bad)} disagreement(s); first 10:")
        for r in bad[:10]:
            print(f"  {r['clip']}")
            print(f"     reference : {r['ref_top1']}  conf={r['ref_conf']}")
            print(f"     ours(allow): {r['fp32_allow_top1']}  conf={r['fp32_allow_conf']:.6f}")
            print(f"     ours(raw)  : {r['fp32_raw_top1']}  conf={r['fp32_raw_conf']:.6f}")
        print("\nThe pruned FP32 model is bit-identical to the full FP32 model on retained "
              "classes, and the full model already scores 100% on this CSV. A disagreement "
              "here is a wiring bug in this driver, not a model result. Stopping without "
              "attempting a fix.")
        raise SystemExit(1)

    print(f"\nGATE PASSED — FP32 allow-list top-1 = 100.000%.")
    if s["fp16"]["allow_top1_pct"] != 100.0:
        print(f"FP16 allow-list top-1 = {s['fp16']['allow_top1_pct']:.3f}% "
              f"(recorded; below 100% does not stop the run).")
    else:
        print("FP16 allow-list top-1 = 100.000% as well.")
    print(f"\nStage 3 order: {', '.join(order)}")


# --------------------------------------------------------------------------- #
# Stages 3, 4, 5
# --------------------------------------------------------------------------- #
def stages345(inv, ref, sessions, sp, sp2idx, allow_idx, only=None) -> None:
    passing = sorted(inv["passing"], key=lambda r: inv["recorders"][r]["scoreable"])
    if only:
        unknown = [r for r in only if r not in passing]
        if unknown:
            raise SystemExit(
                f"FATAL: --only names {unknown}, which did not clear Stage 1. "
                f"Passing recorders: {passing}"
            )
        order = [r for r in passing if r in only]
        carried = [r for r in passing if r not in only]
        print(f"\n=== STAGE 3: scoring only {', '.join(order)} ===")
        if carried:
            print(f"Pass A results carried forward unchanged for: {', '.join(carried)}")
    else:
        order = passing
        carried = []
        print(f"\n=== STAGE 3: per-recorder runs, ascending scoreable ===")
    print("Order: " + ", ".join(f"{r}({inv['recorders'][r]['scoreable']})" for r in order))

    acc = Stage5Acc()
    per_rec: Dict[str, dict] = {}
    all_rows_meta: Dict[str, List[dict]] = {}
    t_start = time.perf_counter()

    for rec in order:
        clips = clips_for(inv, rec, "scoreable_names")
        print(f"\n--- {rec}: {len(clips)} clips ---", flush=True)
        rows, el = score_recorder(clips, ref, sessions, sp, sp2idx, allow_idx, rec, acc)

        detail = ROOT / f"phase4_detail_{rec}.csv"
        write_detail(rows, detail)
        res = {
            "recorder": rec,
            "clips_scored": len(rows),
            "date_coverage": date_coverage(rows),
            "seconds": el,
            "detail_csv": detail.name,
            "models": {t: summarize(rows, t) for t in MODELS},
            "det_sets_identical": sum(r["det_sets_identical"] for r in rows),
        }
        with open(ROOT / f"phase4_results_{rec}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        per_rec[rec] = res
        all_rows_meta[rec] = rows

        for t in MODELS:
            m = res["models"][t]
            print(f"  {t.upper()}: allow-list {m['allow_top1_pct']:.3f}%  "
                  f"conf-match {m['conf_match_pct']:.3f}%  raw {m['raw_top1_pct']:.3f}%")
        print(f"  {el:.1f}s ({el / max(len(rows), 1):.3f}s/clip) -> {detail.name}", flush=True)

    total_s = time.perf_counter() - t_start

    # ---- Carry forward Pass A -------------------------------------------------
    prev_stage5 = None
    if carried:
        prev_path = ROOT / "phase4_results_all_recorders.json"
        if not prev_path.exists():
            raise SystemExit(
                f"FATAL: --only asked to carry forward {carried} but {prev_path.name} "
                f"does not exist. Run a full pass first."
            )
        with open(prev_path, encoding="utf-8") as f:
            prev = json.load(f)
        prev_stage5 = prev.get("stage5")
        for rec in carried:
            rp = ROOT / f"phase4_results_{rec}.json"
            if not rp.exists():
                raise SystemExit(f"FATAL: carried recorder {rec} has no {rp.name}.")
            with open(rp, encoding="utf-8") as f:
                per_rec[rec] = json.load(f)
        print(f"\nCarried forward {len(carried)} recorder result(s) from the prior pass.")

    order = sorted(per_rec, key=lambda r: per_rec[r]["clips_scored"])

    # ---- Stage 4 -----------------------------------------------------------
    print("\n" + "=" * 100)
    print("STAGE 4 — AGGREGATE")
    print("=" * 100)
    for t in MODELS:
        print(f"\n[{t.upper()}]  pruned-493 vs birdnet v0.1.7 reference CSV")
        print(f"{'recorder':<15}{'date coverage':<30}{'clips':>8}"
              f"{'allow-list top-1':>18}{'conf-match':>13}{'raw top-1':>12}")
        print("-" * 96)
        tot = agree = cm = raw = 0
        for rec in order:
            r = per_rec[rec]
            m = r["models"][t]
            print(f"{rec:<15}{r['date_coverage']:<30}{m['clips']:>8}"
                  f"{m['allow_top1_pct']:>17.3f}%{m['conf_match_pct']:>12.3f}%"
                  f"{m['raw_top1_pct']:>11.3f}%")
            tot += m["clips"]; agree += m["allow_top1_n"]
            cm += m["conf_match_n"]; raw += m["raw_top1_n"]
        print("-" * 96)
        print(f"{'TOTAL':<15}{'':<30}{tot:>8}{100 * agree / tot:>17.3f}%"
              f"{100 * cm / tot:>12.3f}%{100 * raw / tot:>11.3f}%")

    excluded = []
    for rec, r in inv["recorders"].items():
        if rec in order:
            continue
        why = "; ".join(r["hard_stops"]) if r["hard_stops"] else (
            "no files on disk (archives unresolved; deferred to Pass B)"
            if r["absent"] else "not scored")
        excluded.append((rec, why, r["csv_rows"]))
    if excluded:
        print("\nExcluded from the tables above (not scored):")
        for rec, why, nrows in sorted(excluded):
            print(f"  {rec}: {why}  [{nrows} CSV rows unrepresented]")

    print("\nRaw top-1 is the unrestricted argmax over all 493 pruned classes, compared "
          "against a reference whose own predictions were filtered to the 218 allow-list "
          "species; the two quantities are not measuring the same thing.")
    print(f"\nTotal Stage 3 wall-clock: {total_s / 60:.1f} min for {sum(per_rec[r]['clips_scored'] for r in order)} clips")

    # ---- Stage 5 -----------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"STAGE 5 — FP16 vs FP32 DETECTION SETS AT {DET_THRESHOLD}")
    print("=" * 100)
    print(f"Threshold {DET_THRESHOLD} is specific to this stage; the existing driver's "
          f"MIN_CONF of 0.1 is not reused.")
    if prev_stage5:
        acc.merge_prior(prev_stage5)
        print(f"(merged with the prior pass: totals below cover all scored recorders)")
    ndiff = acc.n - acc.identical
    print(f"\nClips compared:            {acc.n}")
    print(f"Identical detection sets:  {acc.identical} ({100 * acc.identical / max(acc.n, 1):.4f}%)")
    print(f"Clips with a difference:   {ndiff}")
    if acc.diffs:
        print(f"\nAll {len(acc.diffs)} differing (clip, species) pairs:")
        for d in acc.diffs:
            side = "FP32 only" if d["in_fp32_set"] else "FP16 only"
            print(f"  {d['clip']}  {d['species']}  [{side}]  "
                  f"fp32={d['fp32_conf']:.6f} fp16={d['fp16_conf']:.6f} "
                  f"delta={d['delta_fp16_minus_fp32']:+.2e}")
    if acc.max_dev_at:
        nm, spname = acc.max_dev_at
        print(f"\nMax |FP32-FP16| confidence deviation (all 493 classes, all clips): "
              f"{acc.max_dev:.3e}")
        print(f"  at {nm}  species={spname}")
    if acc.closest_at:
        nm, spname, tag, val = acc.closest_at
        print(f"Closest approach to the {DET_THRESHOLD} boundary by any class in either model: "
              f"{acc.closest:.3e}")
        print(f"  at {nm}  species={spname}  model={tag.upper()}  conf={val:.9f}")
        if acc.max_dev < acc.closest:
            print(f"\n  The largest confidence deviation ({acc.max_dev:.3e}) is smaller than the "
                  f"closest any class came to the boundary ({acc.closest:.3e}), so no class "
                  f"could have crossed. Zero set differences is arithmetically forced here, "
                  f"not merely observed.")

    stage5 = {
        "threshold": DET_THRESHOLD,
        "clips_compared": acc.n,
        "identical_sets": acc.identical,
        "differing_clips": ndiff,
        "differences": acc.diffs,
        "max_abs_conf_deviation": acc.max_dev,
        "max_abs_conf_deviation_at": (
            {"clip": acc.max_dev_at[0], "species": acc.max_dev_at[1]}
            if acc.max_dev_at else None),
        "closest_to_threshold": acc.closest,
        "closest_to_threshold_at": (
            {"clip": acc.closest_at[0], "species": acc.closest_at[1],
             "model": acc.closest_at[2], "conf": acc.closest_at[3]}
            if acc.closest_at else None),
    }

    agg = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {"fp32": str(FP32_MODEL), "fp16": str(FP16_MODEL)},
        "labels": str(FP32_LABELS),
        "classes": EXPECTED_CLASSES,
        "allowlist_size": EXPECTED_ALLOW,
        "conf_tol": CONF_TOL,
        "threads": {"intra_op": 1, "inter_op": 1},
        "scored_recorders": order,
        "excluded": [{"recorder": r, "reason": w, "csv_rows": n} for r, w, n in excluded],
        "per_recorder": per_rec,
        "totals": {
            t: {
                "clips": sum(per_rec[r]["models"][t]["clips"] for r in order),
                "allow_top1_pct": 100 * sum(per_rec[r]["models"][t]["allow_top1_n"] for r in order)
                / sum(per_rec[r]["models"][t]["clips"] for r in order),
                "conf_match_pct": 100 * sum(per_rec[r]["models"][t]["conf_match_n"] for r in order)
                / sum(per_rec[r]["models"][t]["clips"] for r in order),
                "raw_top1_pct": 100 * sum(per_rec[r]["models"][t]["raw_top1_n"] for r in order)
                / sum(per_rec[r]["models"][t]["clips"] for r in order),
            } for t in MODELS
        },
        "stage5": stage5,
        "total_seconds": total_s,
    }
    with open(ROOT / "phase4_results_all_recorders.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregate -> phase4_results_all_recorders.json")
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Stage 6
# --------------------------------------------------------------------------- #
def stage6(inv, sessions, sp, allow_idx, only=None) -> None:
    order = sorted(inv["passing"], key=lambda r: inv["recorders"][r]["scoreable"])
    if only:
        order = [r for r in order if r in only]
    allow_species = sp[allow_idx]
    print("\n" + "=" * 100)
    print("STAGE 6 — NEGATIVE CHECK ON NON-CSV CLIPS")
    print("=" * 100)
    print(f"Seed: {NEG_SEED}   target sample per recorder: {NEG_SAMPLE}   "
          f"threshold: {DET_THRESHOLD}")

    out = {"seed": NEG_SEED, "threshold": DET_THRESHOLD, "target_per_recorder": NEG_SAMPLE,
           "recorders": {}}
    neg_path = ROOT / "phase4_negative_check.json"
    if only and neg_path.exists():
        with open(neg_path, encoding="utf-8") as f:
            out["recorders"] = json.load(f).get("recorders", {})
        print(f"Carrying forward prior results for: "
              f"{', '.join(r for r in out['recorders'] if r not in order)}")
    total_det = 0

    for rec in order:
        pool = inv["recorders"][rec]["non_csv_pool_names"]
        rng = random.Random(NEG_SEED)
        if len(pool) <= NEG_SAMPLE:
            sample = sorted(pool)
            note = f"pool has only {len(pool)} clips; all sampled"
        else:
            sample = sorted(rng.sample(sorted(pool), NEG_SAMPLE))
            note = f"sampled {NEG_SAMPLE} of {len(pool)}"
        print(f"\n--- {rec}: {note} ---", flush=True)
        if not sample:
            out["recorders"][rec] = {"sampled": 0, "note": "empty pool", "detections": []}
            print("  empty pool; skipped")
            continue

        idx = inv["index"]
        clips = [(nm, Path(idx[nm])) for nm in sample]
        dets = []
        n = 0
        t0 = time.perf_counter()
        for nm, probs, _rms in iter_scored(clips, sessions, f"neg/{rec}"):
            n += 1
            for tag in MODELS:
                p = probs[tag][allow_idx]
                hits = np.nonzero(p >= DET_THRESHOLD)[0]
                for h in hits:
                    dets.append({"clip": nm, "model": tag,
                                 "species": str(allow_species[h]),
                                 "conf": float(p[h])})
        el = time.perf_counter() - t0
        out["recorders"][rec] = {"sampled": n, "pool_size": len(pool), "note": note,
                                 "detections": dets, "seconds": el}
        total_det += len(dets)
        print(f"  scored {n} clips in {el:.1f}s -> "
              f"{len(dets)} allow-list detection(s) >= {DET_THRESHOLD}")
        for d in dets:
            print(f"     {d['clip']}  {d['model'].upper()}  {d['species']}  conf={d['conf']:.6f}")

    print("\n" + "-" * 100)
    total_det = sum(len(r["detections"]) for r in out["recorders"].values())
    print("Across all recorders on record:")
    for rec, r in sorted(out["recorders"].items()):
        print(f"  {rec}: {len(r['detections'])} detection(s) on {r['sampled']} sampled clips")
    if total_det == 0:
        print(f"No allow-list detections at or above {DET_THRESHOLD} on any sampled non-CSV "
              f"clip under either model. This is the expected result.")
    else:
        print(f"{total_det} detection(s) found, listed above. These are reported on their own "
              f"and are NOT folded into any parity percentage.")
    with open(ROOT / "phase4_negative_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Negative check -> phase4_negative_check.json")
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 dual-model (FP32+FP16) coverage")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate", action="store_true", help="Stage 2 only")
    g.add_argument("--run", action="store_true", help="Stages 3, 4, 5")
    g.add_argument("--negative", action="store_true", help="Stage 6")
    ap.add_argument("--only", default=None,
                    help="Comma-separated recorders to score; others are carried "
                         "forward from the prior pass rather than recomputed")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",")] if args.only else None

    sp, sp2idx, allow_idx, _species_list = assert_wiring()
    sessions = build_sessions()
    inv = load_inventory()

    if args.negative:
        stage6(inv, sessions, sp, allow_idx, only)
        return

    ref = load_ref()
    print(f"Reference CSV: {len(ref)} distinct clip_name")
    if args.gate:
        stage2(inv, ref, sessions, sp, sp2idx, allow_idx)
    else:
        stages345(inv, ref, sessions, sp, sp2idx, allow_idx, only)


if __name__ == "__main__":
    main()
