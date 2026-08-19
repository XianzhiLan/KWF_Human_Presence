"""Central path configuration.

Every path the pipeline touches resolves through here, driven by environment
variables with repo-relative defaults. Nothing outside this module hardcodes an
absolute path, so the scripts run on any machine without editing.

    BIRDNET_WORK_DIR      scratch dir for non-tracked inputs/outputs (default: repo root)
    BIRDNET_AUDIO_ROOT    root of the 3-second 48 kHz WAV clips
    BIRDNET_AUDIO_EXTRA   extra clip roots, os.pathsep-separated
    BIRDNET_REFERENCE_CSV the birdnet v0.1.7 reference predictions CSV
    BIRDNET_SAVEDMODEL    the upstream BirdNET v2.4 audio-model SavedModel
    BIRDNET_ARCHIVE_DIR   where recorder .zip archives live

The audio and the upstream SavedModel are not in this repo; see README.md. The
reference CSV and species allow-lists ARE in this repo, so they resolve with no
environment variable needed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent


def _p(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return Path(v).expanduser() if v else default


# --- Directories -------------------------------------------------------------
WORK_DIR = _p("BIRDNET_WORK_DIR", REPO_ROOT)
MODELS_DIR = REPO_ROOT / "models"
VERIFICATION_DATA = REPO_ROOT / "verification_data"
ARCHIVE_DIR = _p("BIRDNET_ARCHIVE_DIR", WORK_DIR / "archives")

# --- Inputs not tracked in this repo ------------------------------------------
AUDIO_ROOT = _p("BIRDNET_AUDIO_ROOT", WORK_DIR / "3s48Hz clips")
REFERENCE_CSV = _p(
    "BIRDNET_REFERENCE_CSV",
    VERIFICATION_DATA / "birdnet_species_108k_fp16_verification.csv",
)
SAVEDMODEL = _p("BIRDNET_SAVEDMODEL", WORK_DIR / "BirdNET_v2.4_protobuf" / "audio-model")
SAVEDMODEL_PATCHED = SAVEDMODEL.parent / "audio-model-dft-patched"
EN_US_LABELS = SAVEDMODEL.parent / "labels" / "en_us.txt"

# --- Non-repo full-size models (not tracked; see README) ---------------------
FP32_ONNX = WORK_DIR / "birdnet_fp32.onnx"
FP16_ONNX = WORK_DIR / "birdnet_fp16.onnx"


def audio_roots() -> List[Path]:
    """AUDIO_ROOT plus any BIRDNET_AUDIO_EXTRA entries that exist on disk."""
    roots = [AUDIO_ROOT]
    extra = os.environ.get("BIRDNET_AUDIO_EXTRA", "")
    roots += [Path(p) for p in extra.split(os.pathsep) if p.strip()]
    return [r for r in roots if r.is_dir()]


# --- Pruned models and their tracked sidecars (all IN this repo) -------------
PRUNED_FP32_ONNX = MODELS_DIR / "birdnet_fp32_pruned493.onnx"
PRUNED_FP16_ONNX = MODELS_DIR / "birdnet_fp16_pruned493.onnx"
PRUNED_FP32_LABELS = MODELS_DIR / "birdnet_fp32_pruned493.labels.txt"
PRUNED_FP16_LABELS = MODELS_DIR / "birdnet_fp16_pruned493.labels.txt"

# --- Allow-lists (both IN this repo; NOT the same derivation -- see
# derive_geomodel_allowlist.py and build_allowlist.py docstrings) -------------
ALLOWLIST = VERIFICATION_DATA / "reference_csv_species_allowlist.txt"
PRUNING_KEEPSET_CSV = VERIFICATION_DATA / "pruning_keepset_493.csv"
GEOMODEL_ALLOWLIST = MODELS_DIR / "geomodel_allowlist_218.txt"

# --- Generated outputs ---------------------------------------------------------
INVENTORY_JSON = WORK_DIR / "preflight_inventory.json"


def require(path: Path, what: str) -> Path:
    """Fail with an actionable message instead of a bare FileNotFoundError."""
    if not path.exists():
        raise SystemExit(
            f"Missing {what}: {path}\n"
            f"Set the matching BIRDNET_* environment variable, or see README.md "
            f"for how to obtain it. This file is not tracked in the repo."
        )
    return path
