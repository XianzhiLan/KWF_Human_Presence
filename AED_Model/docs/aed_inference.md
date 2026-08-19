# TinyCNN Audio Inference Module

## Overview

This module is used to run inference with a trained TinyCNN model on a folder of audio clips. It can be used in two ways:

1. **As a Python module**, where other scripts can import the inference functions.
2. **As a standalone script**, where a folder of clips is processed and predictions are saved to a CSV file.

Keeping the core inference functions separate from the command-line interface makes the code easier to reuse in future projects or analysis scripts.

---

# Module Structure

The module is divided into several main functions, each responsible for one part of the inference process.

## `load_model()`

Loads a trained TinyCNN model from a checkpoint file and prepares it for inference.

This function also reads saved model information, such as:

- model version
- training notes
- evaluation metrics (if available)

Arguments:
- model_path: Path to a .pth checkpoint containing 'model_state_dict' and (optionally) 'notes', 'val_acc', 'not_meaningful_recall','not_meaningful_f1'.
- device: torch.device to load the model onto. If None, picks cuda, then mps, then cpu (whichever is available first).

Returns:
- (model, checkpoint) where model is a TinyCNN in eval() mode already moved to `device`, and checkpoint is the raw loaded dict (so callers
can read any metadata it contains).

---

## `preprocess_clip()`

Preprocesses a single audio clip into the format expected by the model.

This ensures every clip is processed in the same way as during training.

The preprocessing pipeline includes:

1. Loading the audio using `librosa`.
2. Resampling audio to **48 kHz**.
3. Converting audio to **mono**.
4. Keeping only the first **3 seconds** of audio.
5. Padding shorter clips with zeros.
6. Generating a mel spectrogram with:
   - 128 mel bands
   - FFT size (`n_fft`) = 1024
   - Hop length = 512
   - Frequency range = 50–16,000 Hz
7. Converting the spectrogram to the decibel scale.
8. Keeping the first 256 time frames.
9. Converting the data to `float32`.
10. Reshaping into the format expected by TinyCNN.

The final input tensor has the shape:

```
(B, 1, 128, 256)
```

where:

- `B` = batch size
- `1` = audio channel
- `128` = mel frequency bins
- `256` = time frames

---

## `predict()`

Runs the TinyCNN model on one or more preprocessed audio clips.

Arguments:
- model: a loaded TinyCNN (eval mode).
- mel_batch: tensor of shape (B, 1, N_MELS, N_FRAMES).
- device: torch.device to run inference on.

Returns:

- prediction probabilities from the model

---

## `score_folder()`

This is the main inference function of the module.

It:

1. Finds supported audio files in a folder.
2. Preprocesses each clip.
3. Runs inference in batches.
4. Converts prediction probabilities into labels.
5. Saves predictions to a CSV file.
6. Returns predictions as a pandas DataFrame.

Arguments:
- folder_path: Directory containing audio clips to score.
- model_path: Path to the .pth checkpoint file.
- output_csv: Full path (including filename) to write the CSV to.
- recursive: If True, search subfolders too.
- audio_extensions: Tuple of file extensions to include.
- batch_size: Batch size for inference.
- threshold: Probability cutoff for predicted_label. A clip is
            labeled "not_meaningful" if not_meaningful_prob >= threshold,
            else "meaningful".
- device: torch.device to run on. If None, auto-detected.

Returns:
- pd.DataFrame of the same rows written to output_csv.

Because `score_folder()` returns a DataFrame, other scripts can directly use the prediction results without needing to read the CSV file again.

Output CSV columns:

| Column | Description |
|---|---|
| `clip_name` | Name of the audio clip |
| `audio_path` | Path to the original audio file |
| `not_meaningful_prob` | Model prediction probability |
| `predicted_label` | Predicted class after applying the threshold |
| `threshold` | Threshold used for classification |
| `model_version` | Model version or checkpoint information |

---

# Inference Workflow

The complete inference workflow is:

1. Load the trained TinyCNN model.
2. Search the input folder for audio clips.
3. Preprocess each audio clip.
4. Run inference on batches of clips.
5. Convert probabilities into predicted labels using a threshold.
6. Save the prediction results to a CSV file.

The default classification threshold is:

```
0.5
```

However, the threshold can be adjusted depending on the desired precision/recall tradeoff.

---

# Output

The output CSV contains one row for each processed audio clip.

| Column | Description |
|---|---|
| `clip_name` | Name of the audio clip |
| `audio_path` | Path to the original audio file |
| `not_meaningful_prob` | Model prediction probability |
| `predicted_label` | Predicted class after applying the threshold |
| `threshold` | Threshold used for classification |
| `model_version` | Model version or checkpoint information |

---

# Running the Module

## Option 1: Use as a Python Module

The inference function can be imported into another Python script.

```python
from aed_inference import score_folder

results = score_folder(
    folder_path="audio/",
    model_path="outputs/models/tinycnn_v2.pth",
    output_csv="outputs/predictions.csv"
)

print(results.head())
```

The returned object is a pandas DataFrame containing all predictions.

---

## Option 2: Run from the Command Line

Basic usage:

```bash
python aed_inference.py \
    --input_folder audio/ \
    --model_path outputs/models/tinycnn_v2.pth \
    --output_csv outputs/predictions.csv
```

---

# Command-Line Arguments

## `--input_folder`

Path to the folder containing audio clips.

Example:

```bash
--input_folder unknown_clips/
```

By default, the script searches recursively through all subfolders.

---

## `--model_path`

Path to the trained TinyCNN checkpoint.

Example:

```bash
--model_path outputs/models/tinycnn_v2.pth
```

---

## `--output_csv`

Path where the prediction CSV will be saved.

Example:

```bash
--output_csv outputs/inference_results.csv
```

---

## `--batch_size`

Controls how many audio clips are processed at once.

Default:

```
64
```

Example:

```bash
--batch_size 128
```

A larger batch size can improve inference speed but requires more memory.

---

## `--threshold`

Sets the probability cutoff for converting model scores into predicted labels.

Default:

```
0.5
```

Example:

```bash
--threshold 0.9
```

Example:

```
Probability = 0.95
Threshold = 0.90

Prediction = 1
```

```
Probability = 0.60
Threshold = 0.90

Prediction = 0
```

---

## `--no_recursive`

By default, the script searches all subfolders inside the input folder.

Example folder:

```
audio/
├── clip1.wav
├── clip2.wav
└── recorder_A/
    └── clip3.wav
```

Default behavior:

```
Processes clip1.wav
Processes clip2.wav
Processes clip3.wav
```

To only process files directly inside the input folder:

```bash
--no_recursive
```

Result:

```
Processes clip1.wav
Processes clip2.wav

Ignores recorder_A/clip3.wav
```

---

# Example Full Command

Example running inference on an unknown audio dataset:

```bash
python aed_inference.py \
    --input_folder data/unknown_clips \
    --model_path outputs/models/tinycnn_v2.pth \
    --output_csv outputs/inference_v2.csv \
    --batch_size 128 \
    --threshold 0.9
```

This command:

- loads TinyCNN v2,
- processes all audio clips in `data/unknown_clips`,
- uses batches of 128 clips,
- classifies clips using a 0.9 probability threshold,
- saves predictions to `outputs/inference_v2.csv`.

---

# Notes

- The model checkpoint must match the preprocessing pipeline used during training.
- Changing the threshold changes the precision/recall tradeoff.
- For large datasets, increasing `batch_size` can reduce inference time if enough memory is available.
- The CSV output can be used directly for downstream analysis, auditing, or additional labeling.
