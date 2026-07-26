# Feature Pipeline — Progress Summary

**Author:** Qian Fu  
**Date:** July 20, 2026  
**Project:** KWF Human Presence Model

---

## Step 1 — Positive Clip Dataset

A teammate ran audio-space augmentation (pitch shift, time stretch, Gaussian noise) on all positive clips. Each original clip produced ~40 augmented variants. All clips were downloaded from Google Drive and organized by device into a unified source directory.

File naming convention:
- Original: `Audio_Moth_1_20250318_171418.wav`
- Augmented: `Audio_Moth_1_20250318_171418_aug_0.wav`

| Device | Original Clips | Augmented Clips | Total |
|---|---:|---:|---:|
| Audio_Moth_1 | 901 | 35,000 | 35,901 |
| Audio_Moth_2 | 1,201 | 50,000 | 51,201 |
| Audio_Moth_3 | 500 | 20,000 | 20,500 |
| Audio_Moth_4 | 2,376 | 95,000 | 97,376 |
| Audio_Moth_5 | 2,822 | 115,000 | 117,822 |
| Audio_Moth_6 | 3,071 | 130,711 | 133,782 |
| **Total** | **10,871** | **445,711** | **456,582** |

---

## Step 2 — Train / Test Split

Because augmentation was performed before splitting, a standard random split would cause data leakage — the same original clip could appear in both train and test sets under different augmented names. To prevent this, a **group-based split** was used: each original clip and all its augmented variants are always assigned to the same partition. The **test set contains only original clips** — no augmented clips ever enter the test set.

**Parameters:**
- Method: Group-based stratified split
- Ratio: 80 / 20
- Random seed: 42
- Group key: root clip name (strip `_aug_N` suffix)
- Test set: original clips only

| Device | Train — Orig | Train — Aug | Train Total | Test — Orig Only |
|---|---:|---:|---:|---:|
| Audio_Moth_1 | 720 | 28,194 | 28,914 | 181 |
| Audio_Moth_2 | 958 | 40,071 | 41,029 | 243 |
| Audio_Moth_3 | 399 | 16,146 | 16,545 | 101 |
| Audio_Moth_4 | 1,898 | 75,894 | 77,792 | 478 |
| Audio_Moth_5 | 2,257 | 91,876 | 94,133 | 565 |
| Audio_Moth_6 | 2,456 | 104,553 | 107,009 | 615 |
| **Total** | **8,688** | **356,734** | **365,422** | **2,183** |

Files are linked using OS hardlinks (not copied) — no extra disk space consumed.  
Script: `feature_pipeline/scripts/train_test_split.py`

---

## Step 3 — Acoustic Feature Extraction

Ten acoustic features are extracted from every clip using **librosa 0.11.0**. Each feature is the mean across all frames of the clip, producing one scalar per feature per clip. Extraction runs in parallel across 8 CPU workers.

**Parameters:**

| Parameter | Value |
|---|---|
| Sample rate | 22,050 Hz |
| FFT window (n_fft) | 2,048 |
| Hop length | 512 |
| MFCC coefficients | 13 |
| Workers | 8 (parallel) |

**Features extracted:**

| Column | Description |
|---|---|
| `RMS_Energy` | Root mean square energy — overall loudness |
| `Spectral_Contrast` | Peak vs. valley difference across frequency bands |
| `Spectral_Flatness` | How noise-like vs. tonal the signal is |
| `Spectral_Bandwidth` | Weighted spread of frequencies around the centroid |
| `Spectral_Rolloff_85` | Frequency below which 85% of energy falls |
| `Onset_Strength` | Rate and intensity of new sound events |
| `MFCC_8` | Mel-frequency cepstral coefficient 8 |
| `MFCC_9` | Mel-frequency cepstral coefficient 9 |
| `MFCC_12` | Mel-frequency cepstral coefficient 12 |
| `MFCC_13` | Mel-frequency cepstral coefficient 13 |

**Output files:**

| File | Rows | Columns |
|---|---|---|
| `feature_pipeline/features/train_acoustic_features.csv` | 365,422 | 11 (clip_name + 10 features) |
| `feature_pipeline/features/test_acoustic_features.csv` | 2,183 | 11 (clip_name + 10 features) |

Join key: `clip_name` (file stem without `.wav` extension). For augmented clips, strip `_aug_N` to look up the parent clip's non-acoustic features from the original dataset CSV.  
Script: `feature_pipeline/scripts/extract_acoustic_features.py`
