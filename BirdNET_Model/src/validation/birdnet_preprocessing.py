"""
Reusable preprocessing + label utilities that reproduce birdnet v0.1.7's exact
behavior for the BirdNET v2.4 acoustic model, for use with the ONNX FP32 export
(birdnet_fp32.onnx).

Everything here is a faithful, standalone reimplementation of the relevant
birdnet v0.1.7 code paths (traced from the installed package source):
  - birdnet/utils.py :: load_audio_in_chunks_with_overlap, get_chunks_with_overlap,
                        resample_array, fillup_with_silence, bandpass_signal,
                        flat_sigmoid, get_species_from_file
  - birdnet/audio_based_prediction.py :: predict_species_within_audio_file_core
  - birdnet/models/v2m4/model_v2m4_base.py :: AudioModelBaseV2M4 constants

Model input contract (confirmed Phase 1/2): input "inputs" (N, 144000) float32,
output "scores" (N, 6522) float32 (raw logits). The mel-spectrogram is computed
inside the graph, so preprocessing is only: load -> resample(48k) -> 3s chunk ->
zero-pad tail -> (default) no-op bandpass.

Notes on why this matches birdnet exactly:
  - Audio is read with soundfile using dtype=np.float32, which performs the
    int-PCM -> float [-1, 1) scaling. No additional normalization or DC removal.
  - Resampling uses scipy.signal.resample (FFT method), per chunk, and is a
    strict no-op when the source sample rate already equals 48000.
  - With birdnet's default bandpass params (fmin=0, fmax=15000) equal to the
    model's (sig_fmin=0, sig_fmax=15000), none of the filter branches fire, so
    bandpass reduces to a float32 cast (mathematical no-op). Replicated faithfully
    so non-default params would still match.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import butter, lfilter, resample

# Model constants (birdnet AudioModelBaseV2M4)
SAMPLE_RATE: int = 48_000
CHUNK_SIZE_S: float = 3.0
SIG_FMIN: int = 0
SIG_FMAX: int = 15_000
CHUNK_SAMPLES: int = round(SAMPLE_RATE * CHUNK_SIZE_S)  # 144000
NUM_CLASSES: int = 6522


# --------------------------------------------------------------------------
# Audio loading / resampling / chunking (mirror of birdnet/utils.py)
# --------------------------------------------------------------------------
def _get_chunks_with_overlap(
    total_duration_s: float, chunk_duration_s: float, overlap_duration_s: float
) -> Generator[Tuple[float, float], None, None]:
    """Exact mirror of birdnet.utils.get_chunks_with_overlap."""
    assert total_duration_s > 0
    assert chunk_duration_s > 0
    assert 0 <= overlap_duration_s < chunk_duration_s

    overlap_duration_s = float(overlap_duration_s)
    chunk_duration_s = float(chunk_duration_s)
    total_duration_s = float(total_duration_s)

    step_duration = chunk_duration_s - overlap_duration_s
    start = 0.0
    while True:
        assert start < total_duration_s
        end = start + chunk_duration_s
        if end < total_duration_s:
            yield start, end
        else:
            yield start, total_duration_s
            break
        start += step_duration


def _resample_array(x: npt.NDArray, sample_rate: int, target_sample_rate: int) -> npt.NDArray:
    """Exact mirror of birdnet.utils.resample_array (scipy.signal.resample, FFT method)."""
    assert len(x.shape) == 1
    assert 0 < sample_rate
    assert 0 < target_sample_rate
    if sample_rate == target_sample_rate:
        return x
    target_sample_count = round(len(x) / sample_rate * target_sample_rate)
    x_resampled: npt.NDArray = resample(x, target_sample_count)
    assert x_resampled.dtype == x.dtype
    return x_resampled


def _fillup_with_silence(audio_chunk: npt.NDArray[np.float32], target_length: int) -> npt.NDArray[np.float32]:
    """Exact mirror of birdnet.utils.fillup_with_silence."""
    current_length = len(audio_chunk)
    assert current_length <= target_length
    if current_length == target_length:
        return audio_chunk
    silence = np.zeros(target_length - current_length, dtype=audio_chunk.dtype)
    return np.concatenate((audio_chunk, silence))


def _bandpass_signal(
    audio_signal: npt.NDArray[np.float32], rate: int, fmin: int, fmax: int, new_fmin: int, new_fmax: int
) -> npt.NDArray[np.float32]:
    """Exact mirror of birdnet.utils.bandpass_signal.

    With birdnet defaults (fmin=0, fmax=15000) == (new_fmin=0, new_fmax=15000),
    none of the branches fire and this is just a float32 cast.
    """
    assert rate > 0
    assert fmin >= 0
    assert fmin < fmax
    assert new_fmin >= 0
    assert new_fmin < new_fmax

    nth_order = 5
    nyquist = 0.5 * rate

    if fmin > new_fmin and fmax == new_fmax:  # Highpass
        low = fmin / nyquist
        b, a = butter(nth_order, low, btype="high")
        audio_signal = lfilter(b, a, audio_signal)
    elif fmin == new_fmin and fmax < new_fmax:  # Lowpass
        high = fmax / nyquist
        b, a = butter(nth_order, high, btype="low")
        audio_signal = lfilter(b, a, audio_signal)
    elif fmin > new_fmin and fmax < new_fmax:  # Bandpass
        low = fmin / nyquist
        high = fmax / nyquist
        b, a = butter(nth_order, [low, high], btype="band")
        audio_signal = lfilter(b, a, audio_signal)

    return audio_signal.astype(np.float32)


def iter_chunks(
    audio_path: Union[str, Path],
    *,
    chunk_duration_s: float = CHUNK_SIZE_S,
    overlap_duration_s: float = 0.0,
    target_sample_rate: int = SAMPLE_RATE,
) -> Generator[Tuple[float, float, npt.NDArray[np.float32]], None, None]:
    """Yield (start_s, end_s, chunk_144000) mirroring birdnet's load path.

    Combines load_audio_in_chunks_with_overlap + fillup_with_silence, exactly as
    predict_species_within_audio_file_core does (before the default no-op bandpass).
    """
    audio_path = Path(audio_path)
    assert audio_path.is_file()

    sf_info = sf.info(audio_path)
    if sf_info.channels != 1:
        raise ValueError(f"{audio_path.name}: birdnet requires mono audio (got {sf_info.channels} channels).")

    sample_rate = sf_info.samplerate
    chunk_sample_size = round(target_sample_rate * chunk_duration_s)

    for start, end in _get_chunks_with_overlap(float(sf_info.duration), float(chunk_duration_s), float(overlap_duration_s)):
        start_samples = round(start * sample_rate)
        end_samples = round(end * sample_rate)
        audio, _ = sf.read(audio_path, start=start_samples, stop=end_samples, dtype=np.float32)
        audio = _resample_array(audio, sample_rate, target_sample_rate)
        audio = _fillup_with_silence(audio, chunk_sample_size)
        yield start, end, audio


def preprocess_file(
    audio_path: Union[str, Path],
    *,
    use_bandpass: bool = True,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
) -> npt.NDArray[np.float32]:
    """Return the (N, 144000) float32 batch birdnet v0.1.7 would feed the model.

    Defaults match birdnet's defaults for predict_species_within_audio_file
    (use_bandpass=True, fmin=0, fmax=15000), which is a no-op transform.
    """
    chunks: List[npt.NDArray[np.float32]] = []
    for _start, _end, chunk in iter_chunks(audio_path):
        if use_bandpass:
            chunk = _bandpass_signal(chunk, SAMPLE_RATE, bandpass_fmin, bandpass_fmax, SIG_FMIN, SIG_FMAX)
        chunks.append(chunk)
    batch = np.array(chunks, dtype=np.float32)
    return batch


# --------------------------------------------------------------------------
# Labels + output post-processing (mirror of birdnet)
# --------------------------------------------------------------------------
def load_species(en_us_path: Union[str, Path]) -> List[str]:
    """Mirror of birdnet.utils.get_species_from_file: OrderedSet(splitlines()).

    OrderedSet preserves first-seen order and removes duplicates. For en_us.txt
    (6522 unique lines, verified) this is identical to a plain line read, but we
    replicate the exact construction so index->species can never silently diverge
    from birdnet's internal mapping.
    """
    from ordered_set import OrderedSet

    text = Path(en_us_path).read_text("utf8")
    return list(OrderedSet(text.splitlines()))


def flat_sigmoid(x: npt.NDArray[np.float32], sensitivity: float = -1.0) -> npt.NDArray[np.float32]:
    """Exact mirror of birdnet.utils.flat_sigmoid (default sensitivity -1.0).

    Monotonic, so it does NOT change argmax/top-1; only needed when comparing or
    displaying confidence *values* against the reference CSV.
    """
    return 1.0 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))
