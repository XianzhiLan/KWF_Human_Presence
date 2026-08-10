"""
aed_inference.py

Score audio clips with a trained TinyCNN checkpoint.

Usable two ways:
    1. As a module — import the functions below into notebooks or other
       scripts (load_model, preprocess_clip, predict, score_folder).
    2. As a script:
           python score_clips.py --input_folder path/to/clips \
                                  --model_path path/to/checkpoint.pth \
                                  --output_csv path/to/predictions.csv

-------------------------------------------------------------------------
Preprocessing pipeline (must match training exactly — do not change
without re-validating against the training pipeline):

  1. Load audio with librosa.load(path, sr=SR, duration=DURATION, mono=True)
     -> resampled to SR Hz, truncated to DURATION seconds, forced to mono.
  2. Zero-pad on the right if the loaded clip is shorter than
     SR * DURATION samples (short clips only; loading already truncates
     clips that are too long).
  3. Compute a mel spectrogram (librosa.feature.melspectrogram) with
     N_MELS mel bands, N_FFT window, HOP_LENGTH hop, restricted to
     [FMIN, FMAX] Hz.
  4. Convert power to decibels with librosa.power_to_db(mel, ref=np.max)
     — this is a per-clip max-referenced normalization, NOT a global
     z-score/mean-std normalization. No additional normalization is
     applied.
  5. Truncate the time axis to the first 256 frames ([:, :256]).
  6. Cast to float32 and add a channel dimension -> shape (1, N_MELS, 256)
     per clip. Batched by the DataLoader -> (B, 1, N_MELS, 256), which is
     what TinyCNN expects as input.
-------------------------------------------------------------------------
"""

import argparse
import glob
import os
import sys

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))
from src.model.architecture import TinyCNN

# ---------------------------------------------------------------------------
# Mel spectrogram parameters (must match training — do not change)
# ---------------------------------------------------------------------------
SR = 48000
DURATION = 3.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMIN = 50
FMAX = 16000
N_FRAMES = 256  # time-axis length the model expects, after truncation

DEFAULT_THRESHOLD = 0.5 
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")


# ---------------------------------------------------------------------------
# Preprocessing — reusable on a single clip
# ---------------------------------------------------------------------------
def preprocess_clip(audio_path):
    """
    Load one audio file and convert it to the mel-spectrogram representation
    TinyCNN expects.

    Returns:
        np.ndarray of shape (N_MELS, N_FRAMES), dtype float32.
    """
    y, sr = librosa.load(audio_path, sr=SR, duration=DURATION, mono=True)
    expected_len = int(SR * DURATION)

    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db[:, :N_FRAMES].astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading  
# ---------------------------------------------------------------------------
def load_model(model_path, device=None):
    """
    Load a TinyCNN checkpoint and return the model plus its metadata.

    Args:
        model_path: Path to a .pth checkpoint containing 'model_state_dict'
            and (optionally) 'notes', 'val_acc', 'not_meaningful_recall',
            'not_meaningful_f1'.
        device: torch.device to load the model onto. If None, picks cuda,
            then mps, then cpu (whichever is available first).

    Returns:
        (model, checkpoint) where model is a TinyCNN in eval() mode already
        moved to `device`, and checkpoint is the raw loaded dict (so callers
        can read any metadata it contains).
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    checkpoint = torch.load(model_path, map_location="cpu")

    model = TinyCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    model_version = checkpoint.get("notes", os.path.basename(model_path))

    print(f"Using model: {os.path.basename(model_path)}")
    print(f"  Version/notes:         {model_version}")
    print(f"  Device:                {device}")
    if "val_acc" in checkpoint:
        print(f"  Val accuracy:          {checkpoint['val_acc']:.4f}")
    if "not_meaningful_recall" in checkpoint:
        print(f"  Not_meaningful recall: {checkpoint['not_meaningful_recall']:.3f}")
    if "not_meaningful_f1" in checkpoint:
        print(f"  Not_meaningful F1:     {checkpoint['not_meaningful_f1']:.3f}")
    print()

    return model, checkpoint


# ---------------------------------------------------------------------------
# Prediction — reusable on an already-batched tensor
# ---------------------------------------------------------------------------
def predict(model, mel_batch, device):
    """
    Run TinyCNN on a batch of preprocessed mel spectrograms.

    Args:
        model: a loaded TinyCNN (eval mode).
        mel_batch: tensor of shape (B, 1, N_MELS, N_FRAMES).
        device: torch.device to run inference on.

    Returns:
        np.ndarray of shape (B,) with not_meaningful probabilities in [0, 1].
    """
    with torch.no_grad():
        mel_batch = mel_batch.to(device)
        raw_probs = torch.sigmoid(model(mel_batch).squeeze(1)).cpu().numpy()
    # Model output is the "meaningful" probability; we report the
    # complement, matching the original convention of this pipeline.
    return 1.0 - raw_probs


# ---------------------------------------------------------------------------
# Dataset/DataLoader plumbing for scoring a whole folder
# ---------------------------------------------------------------------------
class ClipFolderDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["audio_path"]

        if not os.path.exists(path):
            print(f"Missing file:\n{path}")
            return None

        try:
            mel = preprocess_clip(path)
            return torch.from_numpy(mel).unsqueeze(0), row["clip_name"], path
        except Exception as e:
            print(f"Could not read:\n{path}")
            print(e)
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    x, clip_names, paths = zip(*batch)
    return torch.stack(x), clip_names, paths


def find_audio_clips(folder_path, recursive=True, audio_extensions=AUDIO_EXTENSIONS):
    """Return a DataFrame with columns [audio_path, clip_name] for a folder."""
    if recursive:
        pattern = os.path.join(folder_path, "**", "*")
        candidates = glob.glob(pattern, recursive=True)
    else:
        candidates = glob.glob(os.path.join(folder_path, "*"))

    audio_paths = [p for p in candidates if p.lower().endswith(audio_extensions)]

    return pd.DataFrame(
        {
            "audio_path": audio_paths,
            "clip_name": [os.path.basename(p) for p in audio_paths],
        }
    )


# ---------------------------------------------------------------------------
# Top-level orchestration — reusable by other scripts, and used by the CLI
# ---------------------------------------------------------------------------
def score_folder(
    folder_path,
    model_path,
    output_csv,
    recursive=True,
    audio_extensions=AUDIO_EXTENSIONS,
    batch_size=64,
    threshold=DEFAULT_THRESHOLD,
    device=None,
):
    """
    Score every audio clip in a folder and write predictions to a CSV.

    Output CSV columns:
        clip_name, audio_path, not_meaningful_prob, predicted_label,
        threshold, model_version

    Args:
        folder_path: Directory containing audio clips to score.
        model_path: Path to the .pth checkpoint file.
        output_csv: Full path (including filename) to write the CSV to.
        recursive: If True, search subfolders too.
        audio_extensions: Tuple of file extensions to include.
        batch_size: Batch size for inference.
        threshold: Probability cutoff for predicted_label. A clip is
            labeled "not_meaningful" if not_meaningful_prob >= threshold,
            else "meaningful".
        device: torch.device to run on. If None, auto-detected.

    Returns:
        pd.DataFrame of the same rows written to output_csv.
    """
    model, checkpoint = load_model(model_path, device=device)
    device = next(model.parameters()).device
    model_version = checkpoint.get("notes", os.path.basename(model_path))

    clips_df = find_audio_clips(folder_path, recursive, audio_extensions)
    print(f"Number of clips found: {len(clips_df)}")
    print()

    loader = DataLoader(
        ClipFolderDataset(clips_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_skip_none,
    )

    print(f"Scoring {len(clips_df):,} clips on {device}...")
    print(f"Batches: {len(loader):,} x {batch_size} clips")
    print()

    results = []
    for batch in tqdm(loader, total=len(loader)):
        if batch is None:
            continue

        x, clip_names, paths = batch
        probs = predict(model, x, device)

        for clip_name, path, prob in zip(clip_names, paths, probs):
            prob = float(prob)
            results.append(
                {
                    "clip_name": clip_name,
                    "audio_path": path,
                    "not_meaningful_prob": prob,
                    "predicted_label": (
                        "not_meaningful" if prob >= threshold else "meaningful"
                    ),
                    "threshold": threshold,
                    "model_version": model_version,
                }
            )

    columns = [
        "clip_name",
        "audio_path",
        "not_meaningful_prob",
        "predicted_label",
        "threshold",
        "model_version",
    ]
    results_df = pd.DataFrame(results, columns=columns)

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    n_skipped = len(clips_df) - len(results_df)
    print(f"\nSaved {len(results_df):,} predictions to {output_csv}")
    print(f"Skipped {n_skipped:,} missing/corrupt clips.")

    return results_df


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Score a folder of audio clips with a trained TinyCNN checkpoint."
    )
    parser.add_argument(
        "--input_folder", required=True, help="Folder containing audio clips."
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the .pth checkpoint file."
    )
    parser.add_argument(
        "--output_csv", required=True, help="Path to write the output CSV to."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Inference batch size."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability cutoff for predicted_label (default: 0.5).",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Only look at top-level files in input_folder (default: recursive).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    score_folder(
        folder_path=args.input_folder,
        model_path=args.model_path,
        output_csv=args.output_csv,
        recursive=not args.no_recursive,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()