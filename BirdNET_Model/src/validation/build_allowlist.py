"""Persist the reference-CSV species allow-list and resolve the 218-vs-204
discrepancy from the Audio_Moth_1 run.

The reference CSV was generated with a species filter; the allow-list is the set
of species that can appear in it. We resolve:
  - how many DISTINCT species strings appear across all CSV rows (counting every
    element if any cell is a multi-species list), vs. only the top-1 element,
  - how many of those map exactly into en_us.txt (the model label set),
  - which strings (if any) do NOT map, so a label-string mismatch can be ruled in/out.
"""
from __future__ import annotations
import ast, csv, sys
from collections import Counter
from pathlib import Path
import birdnet_preprocessing as bp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

CSV_PATH = config.REFERENCE_CSV
EN_US = config.EN_US_LABELS
OUT = config.ALLOWLIST


def parse_cell(c: str):
    try:
        v = ast.literal_eval(c)
        return [str(s).strip() for s in v] if isinstance(v, (list, tuple)) else [str(v).strip()]
    except (ValueError, SyntaxError):
        return [c.strip()]


def main() -> None:
    species_list = bp.load_species(EN_US)
    label_set = set(species_list)

    all_elems = Counter()      # every species string across all cells
    top1_only = Counter()      # only the first element of each cell
    multi_species_rows = 0
    total_rows = 0
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total_rows += 1
            parts = parse_cell(row["species"])
            if len(parts) > 1:
                multi_species_rows += 1
            top1_only[parts[0]] += 1
            for s in parts:
                all_elems[s] += 1

    distinct_all = set(all_elems)
    distinct_top1 = set(top1_only)
    in_label_all = {s for s in distinct_all if s in label_set}
    not_in_label = sorted(distinct_all - label_set)

    print(f"Total CSV rows: {total_rows}")
    print(f"Rows with >1 species in the cell: {multi_species_rows}")
    print(f"Distinct species (all cell elements): {len(distinct_all)}")
    print(f"Distinct species (top-1 element only): {len(distinct_top1)}")
    print(f"Distinct (all) that map into en_us.txt: {len(in_label_all)}")
    print(f"Distinct (all) NOT in en_us.txt: {len(not_in_label)}")
    if not_in_label:
        print("  Non-label strings:")
        for s in not_in_label:
            print(f"    {s!r}")

    # Allow-list = species that can legitimately appear as a CSV prediction AND
    # map to a model index. Use the union of all cell elements intersected with labels.
    allow = sorted(in_label_all, key=lambda s: species_list.index(s))
    OUT.write_text("\n".join(allow), encoding="utf-8")
    print(f"\nWrote allow-list ({len(allow)} species) -> {OUT}")


if __name__ == "__main__":
    main()
