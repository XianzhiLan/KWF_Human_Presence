"""Pass B — selective extraction of the AM3 and AM5 archives.

Extracts ONLY what the run needs, never the whole archive:
  - every member whose basename appears in that recorder's reference-CSV set
  - plus a seeded sample of non-CSV members for the Stage 6 negative check

Destinations are new, recorder-specific directories so nothing can contaminate
the existing `3s48Hz clips` tree. Archives are opened read-only and never modified.

Halts on: a non-empty destination, a missing archive, or zero CSV matches in an
archive (which would mean the wrong archive or a naming mismatch, not an empty result).
"""
from __future__ import annotations

import csv
import random
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = config.WORK_DIR
CSV_PATH = config.REFERENCE_CSV
ARCHIVES = config.ARCHIVE_DIR

NEG_SAMPLE = 2000
NEG_SEED = 20260809

JOBS = [
    ("Audio_Moth_3", ARCHIVES / "Audio_Moth_3_031925_clips.zip", ROOT / "am3_0319"),
    ("Audio_Moth_5", ARCHIVES / "Audio_Moth_5_031925_clips.zip", ROOT / "am5_0319"),
]

RECORDER_RE = re.compile(r"^(Audio_Moth_\d+)_")


def csv_sets() -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nm = row["clip_name"]
            m = RECORDER_RE.match(nm)
            if m:
                out.setdefault(m.group(1), set()).add(nm)
    return out


def main() -> None:
    wanted = csv_sets()

    for rec, archive, dest in JOBS:
        print("=" * 96)
        print(f"{rec}  <-  {archive.name}")
        print("=" * 96)
        if not archive.is_file():
            raise SystemExit(f"FATAL: archive not found: {archive}")
        if dest.exists() and any(dest.iterdir()):
            raise SystemExit(
                f"FATAL: destination {dest} exists and is not empty. Refusing to extract "
                f"into it — a partial prior extraction would silently mix two sets."
            )
        dest.mkdir(parents=True, exist_ok=True)

        csv_set = wanted.get(rec, set())
        print(f"  CSV references {len(csv_set)} clips for {rec}")
        print(f"  archive size: {archive.stat().st_size / 1e9:.2f} GB")

        with zipfile.ZipFile(archive) as zf:
            members = [i for i in zf.infolist()
                       if not i.is_dir() and i.filename.lower().endswith(".wav")]
            print(f"  members (wav): {len(members)}")

            by_base: Dict[str, zipfile.ZipInfo] = {}
            dupes = 0
            for i in members:
                b = Path(i.filename).name
                if b in by_base:
                    dupes += 1
                else:
                    by_base[b] = i
            if dupes:
                print(f"  NOTE: {dupes} duplicate basename(s) inside the archive; kept first")

            matched = sorted(b for b in by_base if b in csv_set)
            if not matched:
                raise SystemExit(
                    f"FATAL: zero CSV matches in {archive.name}. Either this is the wrong "
                    f"archive for {rec} or the member names do not match CSV clip_name."
                )
            non_csv = sorted(b for b in by_base if b not in csv_set)
            rng = random.Random(NEG_SEED)
            neg = (sorted(rng.sample(non_csv, NEG_SAMPLE))
                   if len(non_csv) > NEG_SAMPLE else non_csv)

            print(f"  matched to CSV : {len(matched)} of {len(csv_set)} "
                  f"({100 * len(matched) / len(csv_set):.1f}% of the CSV rows for {rec})")
            print(f"  non-CSV members: {len(non_csv)}  -> sampling {len(neg)} "
                  f"for the negative check (seed {NEG_SEED})")

            todo = matched + neg
            total_bytes = sum(by_base[b].file_size for b in todo)
            print(f"  extracting {len(todo)} member(s), {total_bytes / 1e9:.2f} GB uncompressed",
                  flush=True)

            done = 0
            for b in todo:
                info = by_base[b]
                with zf.open(info) as src, open(dest / b, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                done += 1
                if done % 2000 == 0:
                    print(f"    ...{done}/{len(todo)}", flush=True)

        on_disk = list(dest.glob("*.wav"))
        bytes_out = sum(p.stat().st_size for p in on_disk)
        print(f"  DONE: {len(on_disk)} file(s) in {dest.name}, "
              f"{bytes_out / 1e9:.2f} GB on disk")
        print(f"  members total={len(members)}  matched={len(matched)}  "
              f"extracted={len(on_disk)}  bytes={bytes_out}")
        sys.stdout.flush()

    print("\nPass B extraction complete. Re-run preflight_inventory.py next; it now "
          "walks these directories as additional roots.")


if __name__ == "__main__":
    main()
