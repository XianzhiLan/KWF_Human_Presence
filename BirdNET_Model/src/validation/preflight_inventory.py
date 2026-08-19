"""Stage 1 preflight inventory for the Phase 4 dual-model coverage run.

READ-ONLY. Never scores, never decodes audio, never moves/renames/deletes.

`3s48Hz clips` is a mixed flat directory PLUS an am1_full subfolder, not one
recorder per directory. So the tree is walked recursively, a basename -> path
index is built, and the recorder is parsed from the FILENAME prefix
(Audio_Moth_{N}_), never from the containing directory.

Length is judged by st_size only (no decode):
    288044 = 144000 samples (correct; 288000 bytes PCM16 mono + 44 byte header)
    295170 = 147456 samples (pre-fix)
    286978 = 143360 samples (pre-fix)

Writes preflight_inventory.json so the scorer resolves every path through this
index and never globs at scoring time.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

CLIPS_ROOT = config.AUDIO_ROOT
# Pass B destinations. Kept as separate roots rather than extracted into the tree
# above, so Pass B audio can never contaminate the Pass A set. Absent roots are
# skipped, which keeps this script valid both before and after Pass B.
EXTRA_ROOTS = [
    config.WORK_DIR / "am3_0319",
    config.WORK_DIR / "am5_0319",
] + config.audio_roots()[1:]
CSV_PATH = config.REFERENCE_CSV
OUT_JSON = config.INVENTORY_JSON

CORRECT_LEN = 288_044
BAD_147456 = 295_170
BAD_143360 = 286_978

RECORDER_RE = re.compile(r"^(Audio_Moth_\d+)_")


def recorder_of(basename: str) -> str:
    m = RECORDER_RE.match(basename)
    return m.group(1) if m else "UNPARSEABLE"


def active_roots() -> List[Path]:
    roots = [CLIPS_ROOT] + [r for r in EXTRA_ROOTS if r.is_dir()]
    return roots


def build_index() -> tuple[Dict[str, Path], List[tuple]]:
    """Return (basename -> path, duplicates). Duplicates = [(name, [(path,size),...])]."""
    seen: Dict[str, Path] = {}
    dupes: Dict[str, List[Path]] = defaultdict(list)
    for root in active_roots():
        for p in root.rglob("*.wav"):
            nm = p.name
            if nm in seen:
                if not dupes[nm]:
                    dupes[nm].append(seen[nm])
                dupes[nm].append(p)
            else:
                seen[nm] = p
    dup_report = [
        (nm, [(str(q), q.stat().st_size) for q in paths]) for nm, paths in dupes.items()
    ]
    return seen, dup_report


def load_csv_by_recorder() -> Dict[str, set]:
    """clip_name sets per recorder, recorder parsed from clip_name prefix."""
    out: Dict[str, set] = defaultdict(set)
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nm = row["clip_name"]
            out[recorder_of(nm)].add(nm)
    return out


def main() -> None:
    if not CLIPS_ROOT.is_dir():
        print(f"FATAL: clips root missing: {CLIPS_ROOT}")
        raise SystemExit(2)

    for r in active_roots():
        print(f"Walking {r} ...")
    index, dupes = build_index()
    print(f"Indexed {len(index)} distinct basenames.")

    if dupes:
        print("\n*** HALT: duplicate basenames found at multiple paths ***")
        for nm, entries in dupes:
            print(f"  {nm}")
            for path, size in entries:
                print(f"      {size:>10} bytes  {path}")
        print(
            "\nPath resolution would be ambiguous and silently non-deterministic. "
            "Resolve the duplication before scoring."
        )
        raise SystemExit(1)
    print("Duplicate-basename check: PASS (no basename appears at two paths).")

    csv_by_rec = load_csv_by_recorder()
    csv_total = sum(len(v) for v in csv_by_rec.values())
    print(f"Reference CSV: {csv_total} distinct clip_name across {len(csv_by_rec)} recorders.")

    # ---- Per-recorder census -------------------------------------------------
    disk_by_rec: Dict[str, List[str]] = defaultdict(list)
    for nm in index:
        disk_by_rec[recorder_of(nm)].append(nm)

    recorders = sorted(set(disk_by_rec) | set(csv_by_rec))
    report: Dict[str, dict] = {}

    for rec in recorders:
        names = disk_by_rec.get(rec, [])
        buckets = {"correct": [], "bad_147456": 0, "bad_143360": 0, "other": 0}
        other_examples: List[tuple] = []
        for nm in names:
            size = index[nm].stat().st_size
            if size == CORRECT_LEN:
                buckets["correct"].append(nm)
            elif size == BAD_147456:
                buckets["bad_147456"] += 1
            elif size == BAD_143360:
                buckets["bad_143360"] += 1
            else:
                buckets["other"] += 1
                if len(other_examples) < 5:
                    other_examples.append((nm, size))

        correct_set = set(buckets["correct"])
        csv_names = csv_by_rec.get(rec, set())
        scoreable = sorted(csv_names & correct_set)
        csv_missing = sorted(csv_names - set(names))
        non_csv_pool = sorted(correct_set - csv_names)

        # A recorder with no files at all is ABSENT (not yet extracted / Pass B),
        # which is a different condition from a present-but-contaminated recorder.
        # Only the latter is a hard stop needing a human re-extraction decision.
        absent = len(names) == 0
        hard_stops = []
        if not absent:
            if len(correct_set) == 0:
                hard_stops.append("correct_len == 0")
            if buckets["bad_147456"] + buckets["bad_143360"] > 0:
                hard_stops.append(
                    f"pre-fix audio present (147456={buckets['bad_147456']}, "
                    f"143360={buckets['bad_143360']})"
                )
            if len(scoreable) == 0:
                hard_stops.append("scoreable == 0")

        report[rec] = {
            "files": len(names),
            "correct_len": len(correct_set),
            "bad_147456": buckets["bad_147456"],
            "bad_143360": buckets["bad_143360"],
            "other": buckets["other"],
            "other_examples": other_examples,
            "csv_rows": len(csv_names),
            "csv_rows_ondisk": len(scoreable),
            "csv_rows_missing": len(csv_missing),
            "non_csv_pool": len(non_csv_pool),
            "scoreable": len(scoreable),
            "absent": absent,
            "hard_stops": hard_stops,
            "scoreable_names": scoreable,
            "non_csv_pool_names": non_csv_pool,
        }

    # ---- Table ---------------------------------------------------------------
    print("\n" + "=" * 108)
    print("STAGE 1 PREFLIGHT INVENTORY")
    print("=" * 108)
    hdr = (
        f"{'recorder':<15}{'files':>9}{'correct_len':>13}{'bad_147456':>12}"
        f"{'bad_143360':>12}{'other':>8}{'csv_rows_ondisk':>17}{'scoreable':>11}"
    )
    print(hdr)
    print("-" * 108)
    for rec in recorders:
        r = report[rec]
        print(
            f"{rec:<15}{r['files']:>9}{r['correct_len']:>13}{r['bad_147456']:>12}"
            f"{r['bad_143360']:>12}{r['other']:>8}{r['csv_rows_ondisk']:>17}"
            f"{r['scoreable']:>11}"
        )
    print("-" * 108)

    print("\nCSV coverage detail (rows referenced but absent from disk):")
    for rec in recorders:
        r = report[rec]
        if r["csv_rows"] == 0:
            continue
        pct = 100 * r["csv_rows_missing"] / r["csv_rows"]
        print(
            f"  {rec:<15} csv_rows={r['csv_rows']:>7}  on_disk={r['csv_rows_ondisk']:>7}  "
            f"absent={r['csv_rows_missing']:>7} ({pct:5.1f}%)   "
            f"non_csv_pool(stage6)={r['non_csv_pool']:>7}"
        )

    any_other = [r for r in recorders if report[r]["other"]]
    if any_other:
        print("\n'other' size bucket detail (not a hard stop; excluded from scoring):")
        for rec in any_other:
            r = report[rec]
            print(f"  {rec}: {r['other']} file(s)")
            for nm, size in r["other_examples"]:
                print(f"      {size:>10} bytes  {nm}")

    # ---- Hard stops ----------------------------------------------------------
    failed = [rec for rec in recorders if report[rec]["hard_stops"]]
    absent = [rec for rec in recorders if report[rec]["absent"]]
    print("\n" + "=" * 108)
    if absent:
        print(
            "ABSENT (no files on disk; deferred to Pass B, not a contamination stop): "
            + ", ".join(absent)
        )
    if failed:
        print("HARD STOP — the following recorders did not clear Stage 1:")
        for rec in failed:
            print(f"  {rec}: {'; '.join(report[rec]['hard_stops'])}")
        print(
            "\nA recorder with pre-fix audio is stale or mixed. This needs a human "
            "decision about re-extraction; it is NOT filtered down and continued."
        )
    passing = [
        rec for rec in recorders
        if not report[rec]["hard_stops"] and not report[rec]["absent"]
    ]
    print(f"\nRecorders clearing Stage 1: {', '.join(passing) if passing else '(none)'}")
    if passing:
        order = sorted(passing, key=lambda r: report[r]["scoreable"])
        print("Stage 3 order (ascending scoreable):")
        for rec in order:
            print(f"  {rec:<15} scoreable={report[rec]['scoreable']}")
        print(f"Stage 2 gate recorder (smallest passing): {order[0]}")

    # ---- Persist -------------------------------------------------------------
    payload = {
        "clips_root": str(CLIPS_ROOT),
        "roots": [str(r) for r in active_roots()],
        "csv_path": str(CSV_PATH),
        "correct_len_bytes": CORRECT_LEN,
        "index": {nm: str(p) for nm, p in index.items()},
        "recorders": report,
        "passing": passing,
        "failed": failed,
        "absent": absent,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"\nIndex + inventory -> {OUT_JSON}  ({OUT_JSON.stat().st_size / 1e6:.1f} MB)")

    sys.stdout.flush()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
