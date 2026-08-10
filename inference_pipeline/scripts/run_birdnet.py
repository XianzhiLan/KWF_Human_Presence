"""
run_birdnet.py — Compressed BirdNET Inference (ONNX)

Runs the pruned FP16 BirdNET ONNX model on 3-second audio clips.

Model:   birdnet_fp16_pruned493.onnx
Labels:  birdnet_fp16_pruned493.labels.txt  (493 species)
Mapping: birdnet_fp16_pruned493.mapping.csv (pruned index → original index)

Input:   folder of .wav clips (3 seconds, any sample rate — resampled to 48kHz)
Output:  CSV with columns: clip_name, species (list), confidence (list)
"""

import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
SR              = 48000          # BirdNET expects 48kHz
DURATION        = 3.0            # seconds
N_SAMPLES       = int(SR * DURATION)  # 144000
CONFIDENCE_THRESHOLD = 0.25

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "birdnet_fp16_pruned493.onnx"
LABELS_PATH = MODELS_DIR / "birdnet_fp16_pruned493.labels.txt"


# ── Load labels ───────────────────────────────────────────────────────────────
def load_labels(labels_path=LABELS_PATH):
    with open(labels_path, "r") as f:
        return [line.strip() for line in f.readlines()]


# ── Load ONNX model ───────────────────────────────────────────────────────────
def load_model(model_path=MODEL_PATH):
    session = ort.InferenceSession(str(model_path))
    return session


# ── Preprocess one clip ───────────────────────────────────────────────────────
def preprocess_clip(audio_path):
    """Load audio, resample to 48kHz, pad/truncate to exactly 144000 samples."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True, duration=DURATION)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]
    return y.astype(np.float32)


# ── Run inference on one clip ─────────────────────────────────────────────────
def run_birdnet_on_clip(audio_path, session, labels, confidence_threshold=CONFIDENCE_THRESHOLD):
    clip_name = Path(audio_path).stem

    try:
        audio = preprocess_clip(audio_path)
        inputs = audio[np.newaxis, :]  # shape: (1, 144000)

        scores = session.run(["scores"], {"inputs": inputs})[0][0]  # shape: (493,)
        probs  = 1 / (1 + np.exp(-scores))  # sigmoid

        species_list    = []
        confidence_list = []
        for i, prob in enumerate(probs):
            if prob >= confidence_threshold:
                species_list.append(labels[i])
                confidence_list.append(round(float(prob), 4))

        return {
            "clip_name":  f"{clip_name}.wav",
            "species":    species_list,
            "confidence": confidence_list,
        }

    except Exception as e:
        print(f"Error processing {Path(audio_path).name}: {e}")
        return {
            "clip_name":  f"{clip_name}.wav",
            "species":    [],
            "confidence": [],
        }


# ── Process all clips in a folder ─────────────────────────────────────────────
def process_all_clips(clips_folder, output_csv, confidence_threshold=CONFIDENCE_THRESHOLD):
    clips_folder = Path(clips_folder)
    clips        = sorted(clips_folder.rglob("*.wav"))

    print(f"Loading model: {MODEL_PATH.name}")
    session = load_model()
    labels  = load_labels()
    print(f"Model loaded. Labels: {len(labels)} species")
    print(f"Processing {len(clips):,} clips from {clips_folder}...")

    rows = []
    for clip in tqdm(clips, desc="BirdNET", unit="clip"):
        row = run_birdnet_on_clip(clip, session, labels, confidence_threshold)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["clip_name", "species", "confidence"])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    detected = (df["species"].apply(len) > 0).sum()
    print(f"\nDone. {len(df):,} clips processed, {detected:,} with detections.")
    print(f"Saved to: {output_csv}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run compressed BirdNET ONNX on audio clips")
    parser.add_argument("--clips_folder", required=True)
    parser.add_argument("--output_csv",   required=True)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    process_all_clips(args.clips_folder, args.output_csv, args.threshold)
