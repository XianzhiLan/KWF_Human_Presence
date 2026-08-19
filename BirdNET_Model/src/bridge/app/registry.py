"""Model registry and load-time safety gates.

The single most dangerous failure mode in this project is pairing a model with
the wrong label file: the model loads, runs, and reports wrong species at full
confidence with no error. Three label files exist and each is valid for exactly
one model:

    en_us.txt                            -> full 6,522-class model only
    birdnet_fp32_pruned493.labels.txt    -> pruned FP32 only
    birdnet_fp16_pruned493.labels.txt    -> pruned FP16 only
    labels.txt (6,362)                   -> nothing, ever

Every gate below runs at startup, not per request. A mispaired configuration
cannot boot the service.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

log = logging.getLogger("bridge.registry")

# Label file that is never valid for any model in this project.
BANNED_LABEL_FILENAMES = {"labels.txt"}
BANNED_LABEL_COUNT = 6362


@dataclass
class LoadedModel:
    variant: str
    model_path: Path
    labels_path: Path
    labels: list[str]
    session: ort.InferenceSession
    input_name: str
    output_name: str
    num_classes: int
    file_mb: float
    intra_op_num_threads: int
    inter_op_num_threads: int
    allow_list: set[str] | None = None
    allow_list_mask: np.ndarray | None = field(default=None, repr=False)

    def allow_list_size(self) -> int | None:
        return None if self.allow_list is None else int(self.allow_list_mask.sum())


class RegistryError(RuntimeError):
    """Raised on any gate failure. Fatal by design — do not catch and continue."""


def _read_labels(path: Path) -> list[str]:
    # utf-8-sig transparently strips a BOM if present and is a no-op otherwise,
    # so this handles both PowerShell's `Set-Content -Encoding utf8` (BOM) and
    # plain utf-8 (no BOM) without needing to know which produced the file.
    with path.open("r", encoding="utf-8-sig") as fh:
        labels = [line.strip() for line in fh if line.strip()]
    return labels


def _gate_label_file_identity(labels_path: Path, labels: list[str]) -> None:
    """Gate 1 — reject the known-bad label file by name and by count."""
    if labels_path.name in BANNED_LABEL_FILENAMES:
        raise RegistryError(
            f"{labels_path.name} is a hand-modified artifact and is not valid for "
            f"any model in this project. Use en_us.txt for the full model or the "
            f"model's own .labels.txt sidecar for a pruned model."
        )
    if len(labels) == BANNED_LABEL_COUNT:
        raise RegistryError(
            f"{labels_path} contains {BANNED_LABEL_COUNT} entries, which matches the "
            f"deprecated labels.txt. Refusing to load."
        )


def _gate_label_count_matches_output(
    labels: list[str], session: ort.InferenceSession, output_name: str, labels_path: Path
) -> int:
    """Gate 2 — label count must equal the model's output dimension.

    This is the check that makes silent mis-indexing impossible.
    """
    out = next(o for o in session.get_outputs() if o.name == output_name)
    dim = out.shape[-1]
    if not isinstance(dim, int):
        # Dynamic output dim: infer it from a single zero-input forward pass.
        inp = session.get_inputs()[0]
        probe = np.zeros((1, 144000), dtype=np.float32)
        dim = int(session.run([output_name], {inp.name: probe})[0].shape[-1])

    if dim != len(labels):
        raise RegistryError(
            f"Label/output mismatch: model emits {dim} classes but {labels_path.name} "
            f"has {len(labels)} entries. These are not a matched pair."
        )
    return int(dim)


def _gate_sidecar_naming(model_path: Path, labels_path: Path) -> None:
    """Gate 3 — a pruned model must be paired with its own sidecar labels file.

    Pruned models are written by prune_birdnet_head.py together with
    <stem>.labels.txt. If the model stem and the labels stem disagree, the pairing
    was made by hand and is suspect.
    """
    stem = model_path.stem
    if "pruned" in stem.lower():
        expected = f"{stem}.labels.txt"
        if labels_path.name != expected:
            raise RegistryError(
                f"Pruned model {model_path.name} must be paired with {expected}, "
                f"got {labels_path.name}."
            )


def _gate_input_shape(session: ort.InferenceSession) -> str:
    """Gate 4 — input must accept (batch, 144000) float32 raw audio."""
    inp = session.get_inputs()[0]
    shape = inp.shape
    if len(shape) != 2 or (isinstance(shape[-1], int) and shape[-1] != 144000):
        raise RegistryError(
            f"Unexpected input shape {shape} on {inp.name}. Expected (dynamic, 144000) "
            f"raw audio — mel transform happens inside the graph."
        )
    if inp.type != "tensor(float)":
        raise RegistryError(f"Expected float32 input, got {inp.type}.")
    return inp.name


def load_variant(
    variant: str,
    spec: dict,
    intra_op_num_threads: int = 1,
    inter_op_num_threads: int = 1,
) -> LoadedModel:
    model_path = Path(spec["model"]).expanduser()
    labels_path = Path(spec["labels"]).expanduser()

    for p in (model_path, labels_path):
        if not p.is_file():
            raise RegistryError(f"Missing file for variant '{variant}': {p}")

    labels = _read_labels(labels_path)
    _gate_label_file_identity(labels_path, labels)
    _gate_sidecar_naming(model_path, labels_path)

    so = ort.SessionOptions()
    # Both default to 1: determinism is the out-of-the-box behavior, not an
    # opt-in. A bridge that returns different numbers run-to-run on the same
    # input is a worse failure than a slow one. Raise via BRIDGE_INTRA_OP_THREADS
    # / BRIDGE_INTER_OP_THREADS for a throughput benchmark, deliberately.
    so.intra_op_num_threads = intra_op_num_threads
    so.inter_op_num_threads = inter_op_num_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(model_path), sess_options=so,
                                   providers=["CPUExecutionProvider"])

    input_name = _gate_input_shape(session)
    output_name = session.get_outputs()[0].name
    num_classes = _gate_label_count_matches_output(labels, session, output_name, labels_path)

    model = LoadedModel(
        variant=variant,
        model_path=model_path,
        labels_path=labels_path,
        labels=labels,
        session=session,
        input_name=input_name,
        output_name=output_name,
        num_classes=num_classes,
        file_mb=round(model_path.stat().st_size / (1024 * 1024), 2),
        intra_op_num_threads=so.intra_op_num_threads,
        inter_op_num_threads=so.inter_op_num_threads,
    )

    allow_path = spec.get("allow_list")
    if allow_path:
        model.allow_list, model.allow_list_mask = _load_allow_list(Path(allow_path), labels)

    log.info(
        "loaded %s: %s (%.2f MB, %d classes) + %s%s",
        variant, model_path.name, model.file_mb, num_classes, labels_path.name,
        f", allow-list {model.allow_list_size()}" if model.allow_list else "",
    )
    return model


def _load_allow_list(path: Path, labels: list[str]) -> tuple[set[str], np.ndarray]:
    """Load the GeoModel allow-list and check containment against the keep-set.

    The allow-list is the 218 species produced by the GeoModel at (8.33, -83.3),
    all-year, min_confidence 0.03. Containment against the model's own label set
    has failed twice on this project, so it is checked every load rather than
    assumed.
    """
    names = {ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}
    label_set = set(labels)
    missing = names - label_set
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise RegistryError(
            f"Containment failure: {len(missing)} of {len(names)} allow-list species are "
            f"absent from this model's keep-set (e.g. {sample}). Scores for these species "
            f"would be silently unobtainable. Rebuild the keep-set before benchmarking."
        )
    mask = np.array([lbl in names for lbl in labels], dtype=bool)
    return names, mask


class Registry:
    def __init__(self) -> None:
        self.models: dict[str, LoadedModel] = {}

    def load_from_config(
        self,
        config_path: Path,
        intra_op_num_threads: int = 1,
        inter_op_num_threads: int = 1,
    ) -> None:
        if not config_path.is_file():
            raise RegistryError(
                f"No models.json at {config_path}. Copy models.json.example and edit "
                f"the paths to point at your ONNX files."
            )
        spec = json.loads(config_path.read_text(encoding="utf-8-sig"))
        variants = spec.get("variants", {})
        if not variants:
            raise RegistryError(f"{config_path} declares no variants.")
        for name, vspec in variants.items():
            self.models[name] = load_variant(
                name, vspec, intra_op_num_threads, inter_op_num_threads
            )

    def get(self, variant: str) -> LoadedModel:
        if variant not in self.models:
            raise KeyError(
                f"Unknown variant '{variant}'. Loaded: {sorted(self.models)}"
            )
        return self.models[variant]

    def clear(self) -> None:
        self.models.clear()
