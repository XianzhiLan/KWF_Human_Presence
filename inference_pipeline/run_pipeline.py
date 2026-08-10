"""
KWF Human Presence — Inference Pipeline
========================================
Master script that runs all steps in order.

Usage:
    python run_pipeline.py --raw_audio /path/to/raw_audio --output /path/to/output

Pipeline steps:
    1. Split raw audio into 3-second clips
    2. Tiny CNN — filter meaningful clips
    3. Compressed BirdNET — species + confidence on meaningful clips only
    4. Acoustic feature extraction
    5. Metadata extraction
    6. Weather data fetch + matching
    7. Feature engineering (sentinel species encoding + Patrick's features)
    8. HP Model inference  [PLACEHOLDER — model not finalized yet]
    9. Output results

Dependencies:
    - PyTorch       (Tiny CNN)
    - onnxruntime   (Compressed BirdNET)  [update run_birdnet.py to use ONNX]
    - librosa       (acoustic features)
    - scikit-learn  (HP Model)
    - requests      (Open-Meteo weather API)
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
SCRIPTS_DIR  = PIPELINE_DIR / "scripts"
MODELS_DIR   = PIPELINE_DIR / "models"
sys.path.insert(0, str(SCRIPTS_DIR))

# ── Script imports ────────────────────────────────────────────────────────────
from audio_split_rename      import split_all_audio
from aed_inference           import score_folder
from run_birdnet             import process_all_clips
from extract_acoustic_features import process_split
from metadata_extraction     import extract_metadata
from weather_info            import get_weather_info_all
from weather_matching        import weather_match
from feature_engineering     import run_feature_engineering

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(PIPELINE_DIR / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step1_split(raw_audio_dir, clips_dir):
    log.info("STEP 1 — Splitting raw audio into 3-second clips...")
    split_all_audio(
        main_folder=str(raw_audio_dir),
        output_folder=str(clips_dir),
    )
    log.info(f"Clips saved to: {clips_dir}")


def step2_filter(clips_dir, outputs_dir):
    """Run Tiny CNN to separate meaningful from non-meaningful clips."""
    log.info("STEP 2 — Tiny CNN: filtering meaningful clips...")
    cnn_csv = outputs_dir / "tinycnn_predictions.csv"
    score_folder(
        folder_path=str(clips_dir),
        model_path=str(MODELS_DIR / "tinycnn_v3.pth"),
        output_csv=str(cnn_csv),
    )

    # Return only meaningful clip names
    df = pd.read_csv(cnn_csv)
    meaningful = df[df["predicted_label"] == "meaningful"]["clip_name"].tolist()
    log.info(f"Meaningful clips: {len(meaningful):,} / {len(df):,} total")
    return meaningful, cnn_csv


def step3_birdnet(clips_dir, meaningful_clips, outputs_dir):
    """
    Run compressed BirdNET on meaningful clips only.

    NOTE: run_birdnet.py currently uses the original TFLite BirdNET library.
    Update run_birdnet.py to use birdnet_fp16_pruned493.onnx (via onnxruntime)
    once the ONNX-compatible script is ready.
    """
    log.info("STEP 3 — Compressed BirdNET: species + confidence...")
    birdnet_csv = outputs_dir / "birdnet_output.csv"

    # Filter folder to meaningful clips only before running BirdNET
    # process_all_clips() will be updated to accept a clip list directly
    process_all_clips(
        clips_folder=str(clips_dir),
        output_csv=str(birdnet_csv),
    )
    log.info(f"BirdNET output saved to: {birdnet_csv}")
    return birdnet_csv


def step4_acoustic(clips_dir, meaningful_clips, outputs_dir):
    """Extract 10 acoustic features from meaningful clips."""
    log.info("STEP 4 — Acoustic feature extraction...")
    acoustic_csv = outputs_dir / "acoustic_features.csv"
    process_split(
        split_name="Inference",
        split_dir=str(clips_dir),
        output_csv=str(acoustic_csv),
    )
    log.info(f"Acoustic features saved to: {acoustic_csv}")
    return acoustic_csv


def step5_metadata(clips_dir, outputs_dir):
    """Extract metadata (clip_name, Recorder, Timestamp, audio properties)."""
    log.info("STEP 5 — Metadata extraction...")
    metadata_csv = outputs_dir / "metadata.csv"
    extract_metadata(
        input_folder=str(clips_dir),
        output_csv=str(metadata_csv),
    )
    log.info(f"Metadata saved to: {metadata_csv}")
    return metadata_csv


def step6_weather(outputs_dir):
    """Fetch live weather data from Open-Meteo API and match to clips."""
    log.info("STEP 6 — Weather data fetch + matching...")
    weather_csv = outputs_dir / "weather.csv"
    get_weather_info_all()
    weather_match()
    log.info(f"Weather data saved to: {weather_csv}")
    return weather_csv


def step7_feature_engineering(acoustic_csv, birdnet_csv, metadata_csv, weather_csv, outputs_dir):
    """Encode sentinel species and engineer Patrick's cross-signal features."""
    log.info("STEP 7 — Feature engineering (sentinel species + Patrick's features)...")
    features_csv = outputs_dir / "features.csv"
    run_feature_engineering(
        acoustic_csv=str(acoustic_csv),
        birdnet_csv=str(birdnet_csv),
        metadata_csv=str(metadata_csv),
        weather_csv=str(weather_csv),
        output_csv=str(features_csv),
    )
    log.info(f"Full feature set saved to: {features_csv}")
    return features_csv


def step8_hp_model(features_csv, outputs_dir):
    """
    Run HP Model (MLP) inference.

    PLACEHOLDER — HP model not finalized yet.
    Once the trained model (.pkl) is available:
        1. Load with: joblib.load("models/hp_model.pkl")
        2. Load scaler with: joblib.load("models/hp_scaler.pkl")
        3. Select the 30 features, scale, and call model.predict_proba()
        4. Apply threshold ~0.38 for final binary prediction
    """
    log.info("STEP 8 — HP Model inference [PLACEHOLDER]...")
    log.warning("HP Model not yet available. Skipping inference.")
    log.info(f"Feature set ready at: {features_csv}")
    log.info("Once the model is finalized, add the inference logic here.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KWF Human Presence Inference Pipeline")
    parser.add_argument("--raw_audio", required=True, help="Folder containing raw AudioMoth recordings")
    parser.add_argument("--output",    required=True, help="Output folder for all intermediate and final results")
    args = parser.parse_args()

    raw_audio_dir = Path(args.raw_audio)
    output_dir    = Path(args.output)
    clips_dir     = output_dir / "clips"
    outputs_dir   = output_dir / "outputs"

    clips_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("KWF Human Presence Inference Pipeline — Starting")
    log.info(f"Raw audio : {raw_audio_dir}")
    log.info(f"Output    : {output_dir}")
    log.info("=" * 60)

    step1_split(raw_audio_dir, clips_dir)
    meaningful_clips, _ = step2_filter(clips_dir, outputs_dir)
    birdnet_csv          = step3_birdnet(clips_dir, meaningful_clips, outputs_dir)
    acoustic_csv         = step4_acoustic(clips_dir, meaningful_clips, outputs_dir)
    metadata_csv         = step5_metadata(clips_dir, outputs_dir)
    weather_csv          = step6_weather(outputs_dir)
    features_csv         = step7_feature_engineering(acoustic_csv, birdnet_csv, metadata_csv, weather_csv, outputs_dir)
    step8_hp_model(features_csv, outputs_dir)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
