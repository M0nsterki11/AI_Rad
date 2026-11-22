"""Local cache identity and opt-in, content-free server model diagnostics."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def weight_path(model_key):
    if model_key == "resnet50":
        from .predict_resnet import active_model_path
        return active_model_path()
    if model_key == "xlm_roberta":
        from .predict_text_model import active_model_dir
    elif model_key == "layoutlmv3":
        from .predict_layoutlm import active_model_dir
    else:
        raise ValueError(f"Unknown model: {model_key}")
    directory = active_model_dir()
    return next((directory / n for n in ["model.safetensors", "pytorch_model.bin"] if (directory / n).is_file()), directory / "model.safetensors")


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def model_cache_identity(model_key):
    """Fingerprint file version metadata on each rerun; no GB reads per upload.

    This detects ordinary replacement/edits, not malicious preservation of all
    file timestamps. Restart explicitly after deployments for that edge case.
    """
    weight = weight_path(model_key)
    files = {weight, *weight.parent.glob("*.json"), *weight.parent.glob("*.model"),
             *weight.parent.glob("*.txt")}
    parent_mapping = weight.parent.parent / "label_mapping.json"
    if parent_mapping.is_file():
        files.add(parent_mapping)
    return tuple(file_identity(p) for p in sorted(files) if p.is_file())


@lru_cache(maxsize=32)
def _sha256(identity):
    digest = hashlib.sha256()
    with Path(identity[0]).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_record(model_key, model=None):
    from .model_downloader import _get_repo_id
    path = weight_path(model_key)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_hashes = {str(p.relative_to(ROOT)): _sha256(file_identity(p)) for p in [ROOT / "app.py", *sorted((ROOT / "src").glob("*.py"))]}
    if model_key == "resnet50":
        mapping = json.loads((path.parent / "label_mapping.json").read_text(encoding="utf-8"))
        id2label = {str(i): label for i, label in enumerate(mapping["class_names"])}
    else:
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        id2label = config.get("id2label")
    return {"event": "document_ai_model_loaded", "code_commit": commit, "source_sha256": source_hashes,
        "model_key": model_key, "resolved_weight_path": str(path.resolve()),
        "weight_sha256": _sha256(file_identity(path)), "weight_bytes": path.stat().st_size,
        "configured_download_repo_id": _get_repo_id(model_key), "download_revision": None,
        "revision_note": "No historical download receipt; configured repo is not proof of file origin",
        "model_in_memory": model is not None, "eval_mode": not model.training if model is not None else None,
        "id2label": id2label,
        "aggregation_method": getattr(model, "document_aggregation_method", None),
        "aggregation_top_k": getattr(model, "document_aggregation_top_k", None)}


def log_loaded_model(model_key, model):
    if os.environ.get("DOCUMENT_AI_MODEL_DIAGNOSTICS") == "1":
        print("DOCUMENT_AI_MODEL " + json.dumps(runtime_record(model_key, model), ensure_ascii=True), flush=True)
