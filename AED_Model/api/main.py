"""
TinyCNN V3 FastAPI inference server.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_PATH  Path to .pth checkpoint (default: outputs/models/tinycnn_v3.pth)
    THRESHOLD   not_meaningful_prob cutoff (default: 0.5)
"""

import io
import os
import sys
from contextlib import asynccontextmanager

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.architecture import TinyCNN

SR = 48000
DURATION = 3.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMIN = 50
FMAX = 16000

MODEL_PATH = os.getenv("MODEL_PATH", "outputs/models/tinycnn_v3.pth")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

_model: TinyCNN | None = None
_device: torch.device | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    _model = TinyCNN()
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _model.to(_device)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"  Device:    {_device}")
    print(f"  Threshold: {THRESHOLD}")
    yield


app = FastAPI(title="TinyCNN Audio Classifier", lifespan=lifespan)


class PredictResponse(BaseModel):
    label: str
    not_meaningful_prob: float


def compute_mel(audio_bytes: bytes) -> torch.Tensor:
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR, duration=DURATION, mono=True)
    expected = int(SR * DURATION)
    if len(y) < expected:
        y = np.pad(y, (0, expected - len(y)))
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)[:, :256].astype(np.float32)
    return torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 256)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted")

    audio_bytes = await file.read()

    try:
        x = compute_mel(audio_bytes).to(_device)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {e}")

    with torch.no_grad():
        prob = torch.sigmoid(_model(x).squeeze(1)).item()

    not_meaningful_prob = round(1.0 - prob, 4)
    label = "not_meaningful" if not_meaningful_prob > THRESHOLD else "meaningful"

    return PredictResponse(label=label, not_meaningful_prob=not_meaningful_prob)


@app.get("/health")
def health():
    return {"status": "ok"}
