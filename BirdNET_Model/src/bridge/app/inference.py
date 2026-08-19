"""Scoring: raw logits -> flat_sigmoid -> threshold -> detections.

The model emits raw logits. The reference pipeline converts them with BirdNET's
flat_sigmoid at sensitivity -1.0, which reduces to a plain logistic on a clipped
input.

INTERNAL_FLOOR
--------------
The reference CSV was generated at threshold 0.25 with an internal 0.1 floor
applied first. That floor is reproduced here so detection sets match the
reference exactly. NOTE: the floor's exact position in the reference call chain
is taken from the project handoff, not from independently re-read birdnet 0.1.7
source. At any user threshold >= 0.1 the floor is a no-op and this is moot; it
only matters if someone benchmarks at min_confidence < 0.1. Confirm before
relying on sub-0.1 results.
"""

from __future__ import annotations

import time

import numpy as np

from .registry import LoadedModel
from .schemas import Detection

INTERNAL_FLOOR = 0.1
SENSITIVITY = -1.0


def flat_sigmoid(x: np.ndarray, sensitivity: float = SENSITIVITY) -> np.ndarray:
    """BirdNET's activation. Clipping to +/-15 is part of the reference behaviour."""
    return 1.0 / (1.0 + np.exp(sensitivity * np.clip(x, -15.0, 15.0)))


def score_window(
    model: LoadedModel,
    window: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run one 144,000-sample window. Returns (confidences, inference_ms).

    Timing brackets session.run only — no preprocessing, no post-processing, no
    allocation of the input batch. This is the number that belongs in a latency
    comparison.
    """
    batch = window.reshape(1, -1).astype(np.float32, copy=False)
    t0 = time.perf_counter()
    logits = model.session.run([model.output_name], {model.input_name: batch})[0]
    inference_ms = (time.perf_counter() - t0) * 1000.0
    return flat_sigmoid(logits[0]), inference_ms


def to_detections(
    model: LoadedModel,
    confidences: np.ndarray,
    data_id: str,
    start_s: float,
    end_s: float,
    min_confidence: float,
    apply_allow_list: bool,
) -> list[Detection]:
    """Threshold and label. Allow-list marking is post-inference, as in production.

    The allow-list never removes classes from the model and never suppresses a
    detection here — it sets in_allow_list=False. Reporting both lets a consumer
    reproduce either the allow-list-filtered numbers (which match the reference
    CSV) or the unrestricted numbers, without a second run. Collapsing the two is
    what made 'raw top-1' look like a regression when it was a different question.
    """
    threshold = max(min_confidence, INTERNAL_FLOOR)
    idx = np.flatnonzero(confidences >= threshold)
    if idx.size == 0:
        return []

    order = idx[np.argsort(-confidences[idx])]
    mask = model.allow_list_mask

    out: list[Detection] = []
    for i in order:
        in_allow = True
        if apply_allow_list and mask is not None:
            in_allow = bool(mask[i])
        out.append(
            Detection(
                data_id=data_id,
                species=model.labels[i],
                confidence=float(confidences[i]),
                start_time=round(start_s, 3),
                end_time=round(end_s, 3),
                in_allow_list=in_allow,
            )
        )
    return out


def warmup(model: LoadedModel, iterations: int = 3) -> None:
    """Burn the first few runs before timing anything.

    ORT's first run pays one-time allocation and kernel-selection cost that is
    several times steady-state latency. Including it in a benchmark makes a model
    look slower than it is, and makes short runs incomparable to long ones.
    """
    dummy = np.zeros(144_000, dtype=np.float32)
    for _ in range(iterations):
        score_window(model, dummy)
