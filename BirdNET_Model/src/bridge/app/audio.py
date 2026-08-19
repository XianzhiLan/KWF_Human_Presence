"""Audio ingestion — the proven bit-exact preprocessing chain.

Reference chain (validated to 0.000e+00 against the birdnet package):

    soundfile.read(dtype="float32")
      -> resample to 48 kHz if needed
      -> split into 144,000-sample windows
      -> no-op bandpass
      -> (model)
      -> flat_sigmoid(sensitivity=-1.0)

flat_sigmoid is applied to the model's raw logits, NOT here — it lives in
inference.py. This module produces float32 windows only.

Wrong-length audio is the pipeline's historical failure mode. 144,000 samples is
288,044 bytes as WAV; 295,170 bytes (147,456 samples) and 286,978 bytes (143,360
samples) are the old pydub millisecond-rounding output and must never be scored.
Short remainders are rejected rather than zero-padded, because padding would
produce a plausible-looking score for a window that the reference pipeline never
evaluated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample as scipy_resample

TARGET_SR = 48_000
WINDOW_SAMPLES = 144_000          # exactly 3.0 s at 48 kHz
WINDOW_SECONDS = WINDOW_SAMPLES / TARGET_SR


class AudioError(ValueError):
    pass


def _resample(x: np.ndarray, sr_in: int) -> np.ndarray:
    """Exact mirror of the driver's _resample_array (scipy.signal.resample, FFT
    method), not a soxr/librosa substitute. Those are different algorithms and
    were never verified to agree with the validated chain at any tolerance;
    matching the driver's own resampler is the point, not merely resampling.

    scipy is imported at module level (not lazily here) so a missing dependency
    fails at import time, immediately and loudly -- not the first time someone
    feeds this a clip that isn't already 48 kHz, possibly in production.
    """
    if sr_in == TARGET_SR:
        return x

    target_sample_count = round(len(x) / sr_in * TARGET_SR)
    return scipy_resample(x, target_sample_count).astype(np.float32, copy=False)


def read_windows(path: Path, strict: bool = True) -> list[tuple[np.ndarray, float, float]]:
    """Read a WAV file into 144,000-sample windows.

    Returns a list of (samples, start_seconds, end_seconds).

    strict=True rejects any file whose length is not an exact multiple of the
    window size. Set False only when deliberately scoring long-form recordings,
    where a trailing partial window is expected and dropped.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)

    if data.ndim > 1:
        # The driver (birdnet_preprocessing.py::iter_chunks) rejects non-mono
        # outright rather than downmixing. Match that exactly: a silent average
        # would be a different signal than the one the reference chain ever
        # scored, not a harmless convenience.
        raise AudioError(
            f"{path.name}: {data.shape[1]} channels. birdnet requires mono audio; "
            f"the reference pipeline never downmixes."
        )

    data = _resample(data, sr)

    n = data.shape[0]
    full = n // WINDOW_SAMPLES
    remainder = n - full * WINDOW_SAMPLES

    if full == 0:
        raise AudioError(
            f"{path.name}: {n} samples is shorter than one {WINDOW_SAMPLES}-sample window."
        )
    if remainder and strict:
        raise AudioError(
            f"{path.name}: {n} samples is not a multiple of {WINDOW_SAMPLES}. "
            f"This is the signature of the old millisecond-rounding split. "
            f"Re-run the fixed split script, or pass strict=false to drop the remainder."
        )

    out = []
    for i in range(full):
        start = i * WINDOW_SAMPLES
        window = data[start:start + WINDOW_SAMPLES]
        # No-op bandpass retained as an explicit step so the chain reads the same
        # as the validated reference. Do not remove; if a real bandpass is ever
        # reintroduced upstream it belongs exactly here.
        out.append((window, i * WINDOW_SECONDS, (i + 1) * WINDOW_SECONDS))
    return out


def collect_paths(paths: list[str], limit: int | None = None) -> list[Path]:
    """Expand a mix of files and directories into a sorted list of WAV paths."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            found.extend(sorted(p.glob("*.wav")))
        elif p.is_file():
            found.append(p)
        else:
            raise AudioError(f"Path does not exist: {p}")
    if not found:
        raise AudioError("No .wav files found in the supplied paths.")
    return found[:limit] if limit else found
