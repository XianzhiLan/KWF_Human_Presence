"""Derive geomodel_allowlist_218.txt, referenced by the bridge's
models.json.example but not materialized as a standalone file until this
script runs.

pruning_keepset_493.csv's in_geo_allowlist column flags exactly the 218
species that came from the GeoModel side of the pruning keep-set's
three-source union (see models/README.md's "What the pruning did"). This is
NOT the same derivation as reference_csv_species_allowlist.txt (built by
build_allowlist.py, from the reference CSV's own predictions) -- the two
happen to overlap heavily but are not guaranteed identical.

    python derive_geomodel_allowlist.py
    python derive_geomodel_allowlist.py --keepset path/to/keepset.csv --out path/to/out.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keepset", type=Path, default=config.PRUNING_KEEPSET_CSV,
                     help="pruning_keepset_493.csv with an in_geo_allowlist column")
    ap.add_argument("--out", type=Path, default=config.GEOMODEL_ALLOWLIST,
                     help="output path for the newline-delimited species list")
    args = ap.parse_args()

    species = []
    with open(args.keepset, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["in_geo_allowlist"] == "1":
                species.append(row["species"])

    if len(species) != 218:
        raise SystemExit(
            f"FATAL: expected 218 species with in_geo_allowlist=1, got {len(species)}. "
            f"Not writing {args.out} with an unexpected count."
        )

    args.out.write_text("\n".join(species), encoding="utf-8")
    print(f"Wrote {len(species)} species -> {args.out}")


if __name__ == "__main__":
    main()
