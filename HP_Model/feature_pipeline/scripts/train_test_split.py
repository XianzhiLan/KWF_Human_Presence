"""
Train/Test Split for HP Model Audio Data

Rules:
- Groups each original clip with all its augmented variants by root filename
- 80/20 stratified split on root filenames (never splits a group across train/test)
- Train set: original clips + all augmented variants
- Test set: original clips ONLY — no augmented clips ever enter the test set
"""

import os
import random
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SOURCE_DIR = Path("/Users/qian/KWF/Human sounds")
OUTPUT_DIR = Path("/Users/qian/KWF/ML engineer team/summer 2026/Models/KWF_Human_Presence/feature_pipeline")
TRAIN_DIR  = OUTPUT_DIR / "train"
TEST_DIR   = OUTPUT_DIR / "test"

# ── Config ──────────────────────────────────────────────────────────────────
TEST_RATIO  = 0.2
RANDOM_SEED = 42
AUDIO_MOTHS = [f"Audio_Moth_{i}" for i in range(1, 7)]

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)

    total_train_orig = 0
    total_train_aug  = 0
    total_test_orig  = 0

    for moth in AUDIO_MOTHS:
        source    = SOURCE_DIR / moth
        train_out = TRAIN_DIR  / moth
        test_out  = TEST_DIR   / moth

        train_out.mkdir(parents=True, exist_ok=True)
        test_out.mkdir(parents=True, exist_ok=True)

        # Group every file by its root clip name
        groups = defaultdict(list)
        for f in source.iterdir():
            if f.suffix == '.wav':
                root = f.name.split('_aug_')[0].replace('.wav', '') if '_aug_' in f.name else f.stem
                groups[root].append(f.name)

        # Shuffle roots and split 80/20
        roots = sorted(groups.keys())
        random.shuffle(roots)

        split_idx   = int(len(roots) * (1 - TEST_RATIO))
        train_roots = set(roots[:split_idx])
        test_roots  = set(roots[split_idx:])

        # Train: originals + augmented
        train_orig = 0
        train_aug  = 0
        for root in train_roots:
            for filename in groups[root]:
                src = source / filename
                dst = train_out / filename
                if not dst.exists():
                    os.link(src, dst)
                if '_aug_' in filename:
                    train_aug += 1
                else:
                    train_orig += 1

        # Test: originals only
        test_orig = 0
        for root in test_roots:
            src = source / (root + '.wav')
            dst = test_out / (root + '.wav')
            if src.exists() and not dst.exists():
                os.link(src, dst)
                test_orig += 1

        total_train_orig += train_orig
        total_train_aug  += train_aug
        total_test_orig  += test_orig

        print(f"{moth}:")
        print(f"  Roots total : {len(roots)}")
        print(f"  Train roots : {len(train_roots)}  →  {train_orig} originals + {train_aug} augmented = {train_orig + train_aug} files")
        print(f"  Test roots  : {len(test_roots)}  →  {test_orig} originals only")
        print()

    print("=" * 55)
    print("TOTAL")
    print(f"  Train : {total_train_orig} originals + {total_train_aug} augmented = {total_train_orig + total_train_aug} files")
    print(f"  Test  : {total_test_orig} originals only")
    print("=" * 55)
    print("Done!")


if __name__ == "__main__":
    main()
