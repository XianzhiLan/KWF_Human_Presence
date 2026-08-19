"""
Acoustic Feature Extraction for HP Model

Extracts 10 acoustic features from each audio clip using librosa.
Processes train and test sets separately and saves to CSV.

Output:
  feature_pipeline/features/train_acoustic_features.csv
  feature_pipeline/features/test_acoustic_features.csv
"""

import os
import csv
import librosa
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/Users/qian/KWF/ML engineer team/summer 2026/Models/KWF_Human_Presence/feature_pipeline")
TRAIN_DIR  = BASE_DIR / "train"
TEST_DIR   = BASE_DIR / "test"
OUTPUT_DIR = BASE_DIR / "features"

# ── Audio parameters ────────────────────────────────────────────────────────
SR         = 22050   # sample rate
N_FFT      = 2048    # FFT window size
HOP_LENGTH = 512     # hop length
N_MFCC     = 13      # number of MFCC coefficients

# ── Output columns ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    "clip_name",
    "RMS_Energy",
    "Spectral_Contrast",
    "Spectral_Flatness",
    "Spectral_Bandwidth",
    "Spectral_Rolloff_85",
    "Onset_Strength",
    "MFCC_8",
    "MFCC_9",
    "MFCC_12",
    "MFCC_13",
]

# ── Feature extraction ───────────────────────────────────────────────────────
def extract_features(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True)

        rms               = float(np.mean(librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)))
        spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)))
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)))
        spectral_bandwidth= float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)))
        spectral_rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85)))
        onset_strength    = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)))
        mfccs             = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

        return {
            "clip_name":           Path(wav_path).stem,
            "RMS_Energy":          rms,
            "Spectral_Contrast":   spectral_contrast,
            "Spectral_Flatness":   spectral_flatness,
            "Spectral_Bandwidth":  spectral_bandwidth,
            "Spectral_Rolloff_85": spectral_rolloff,
            "Onset_Strength":      onset_strength,
            "MFCC_8":              float(np.mean(mfccs[7])),   # 0-indexed: MFCC_8  = index 7
            "MFCC_9":              float(np.mean(mfccs[8])),   # MFCC_9  = index 8
            "MFCC_12":             float(np.mean(mfccs[11])),  # MFCC_12 = index 11
            "MFCC_13":             float(np.mean(mfccs[12])),  # MFCC_13 = index 12
        }

    except Exception as e:
        print(f"\nError processing {Path(wav_path).name}: {e}")
        return None


# ── Process one split ────────────────────────────────────────────────────────
def process_split(split_name, split_dir, output_csv, n_workers=8):
    wav_files = sorted(Path(split_dir).rglob("*.wav"))
    total     = len(wav_files)
    print(f"\n{'='*55}")
    print(f"{split_name} set: {total:,} files")
    print(f"Output: {output_csv}")
    print(f"{'='*55}")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    failed = 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(extract_features, str(p)): p for p in wav_files}

            for future in tqdm(as_completed(futures), total=total, desc=split_name):
                result = future.result()
                if result:
                    writer.writerow(result)
                else:
                    failed += 1

    print(f"Done. {total - failed:,} rows saved, {failed} failed.")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    process_split("Train", TRAIN_DIR, OUTPUT_DIR / "train_acoustic_features.csv")
    process_split("Test",  TEST_DIR,  OUTPUT_DIR / "test_acoustic_features.csv")

    print("\nAll done!")
    print(f"Features saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
