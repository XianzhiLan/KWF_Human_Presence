"""
Feature Engineering Pipeline for HP Model

Takes as input:
  - acoustic_features.csv     (clip_name + 10 librosa features)
  - birdnet_output.csv        (clip_name, species list, confidence list)
  - metadata.csv              (clip_name, Recorder, Timestamp, ...)
  - weather.csv               (clip_name, Temperature, Windspeed, Humidity, ...)

Produces:
  - features.csv              (all 30 features ready for HP model inference)

Features created here:
  - 10 sentinel species binary flags  (from BirdNET species output)
  - Eerie_Silence                     (1 if no sentinel species detected)
  - hour_sin, hour_cos               (cyclical time from Timestamp)
  - Volume_Wind_Ratio                 (RMS / Windspeed)
  - Volume_Spike_15s                  (sudden RMS jump vs. 5-clip rolling mean)
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path

# ── Sentinel species ────────────────────────────────────────────────────────
SENTINEL_SPECIES = [
    'Myiothlypis fulvicauda_Buff-rumped Warbler',
    'Habia atrimaxillaris_Black-cheeked Ant-Tanager',
    'Thamnophilus bridgesi_Black-hooded Antshrike',
    'Tinamus major_Great Tinamou',
    'Patagioenas nigrirostris_Short-billed Pigeon',
    'Ramphastos ambiguus_Yellow-throated Toucan',
    'Cyanoloxia cyanoides_Blue-black Grosbeak',
    'Lipaugus unirufus_Rufous Piha',
    'Threnetes ruckeri_Band-tailed Barbthroat',
    'Ara macao_Scarlet Macaw',
]

# ── Helpers ─────────────────────────────────────────────────────────────────
def safe_eval(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else [result]
    except (ValueError, SyntaxError):
        return []


def encode_sentinel_species(df):
    """Multi-hot encode BirdNET species output into 10 binary sentinel columns."""
    parsed = df['species'].apply(safe_eval)
    for species in SENTINEL_SPECIES:
        df[species] = parsed.apply(lambda sp_list: int(species in sp_list))
    return df


def parse_confidence(val):
    """Extract max confidence from a list or scalar value."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return 0.0
    if isinstance(val, list):
        return float(max(val)) if len(val) > 0 else 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def engineer_patrick_features(df):
    """Create Patrick's 5 cross-signal engineered features."""
    # Cyclical time encoding
    df['Datetime'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')
    df['hour_sin'] = np.sin(df['Datetime'].dt.hour * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['Datetime'].dt.hour * (2 * np.pi / 24))

    # Eerie Silence: 1 if none of the 10 sentinel species detected
    df['Eerie_Silence'] = (df[SENTINEL_SPECIES].sum(axis=1) == 0).astype(int)

    # Volume to Wind Ratio
    df['Volume_Wind_Ratio'] = df['RMS_Energy'] / (df['Windspeed'] + 1e-5)

    # Volume Spike (15-second memory = 5 clips of 3s each)
    df = df.sort_values(by=['Recorder', 'Datetime']).reset_index(drop=True)
    df['rolling_rms'] = (
        df.groupby('Recorder')['RMS_Energy']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df['Volume_Spike_15s'] = (df['RMS_Energy'] - df['rolling_rms']).clip(lower=0)
    df.drop(columns=['rolling_rms'], inplace=True)

    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def run_feature_engineering(
    acoustic_csv,
    birdnet_csv,
    metadata_csv,
    weather_csv,
    output_csv,
):
    print("Loading inputs...")
    acoustic = pd.read_csv(acoustic_csv)
    birdnet   = pd.read_csv(birdnet_csv)
    metadata  = pd.read_csv(metadata_csv)
    weather   = pd.read_csv(weather_csv)

    # Normalize clip_name (strip .wav if present)
    for frame in [acoustic, birdnet, metadata, weather]:
        frame['clip_name'] = frame['clip_name'].str.replace('.wav', '', regex=False)

    print("Merging inputs...")
    df = acoustic.merge(birdnet,  on='clip_name', how='left')
    df = df.merge(metadata[['clip_name', 'Recorder', 'Timestamp']], on='clip_name', how='left')
    df = df.merge(weather[['clip_name', 'Temperature', 'Windspeed', 'Humidity']], on='clip_name', how='left')

    print("Encoding sentinel species...")
    df = encode_sentinel_species(df)

    print("Parsing confidence...")
    df['confidence'] = df['confidence'].apply(parse_confidence)

    print("Engineering Patrick's features...")
    df = engineer_patrick_features(df)

    print("Saving output...")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Done. {len(df):,} rows saved to {output_csv}")

    return df


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent

    run_feature_engineering(
        acoustic_csv = BASE / "outputs" / "acoustic_features.csv",
        birdnet_csv  = BASE / "outputs" / "birdnet_output.csv",
        metadata_csv = BASE / "outputs" / "metadata.csv",
        weather_csv  = BASE / "outputs" / "weather.csv",
        output_csv   = BASE / "outputs" / "features.csv",
    )
