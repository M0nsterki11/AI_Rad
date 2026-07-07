from __future__ import annotations

import os
import shutil
from pathlib import Path


MODEL_REPO_ENV = {
    "resnet50": "HF_RESNET50_REPO_ID",
    "xlm_roberta": "HF_XLM_ROBERTA_REPO_ID",
    "layoutlmv3": "HF_LAYOUTLMV3_REPO_ID",
}
MISSING_REPO_MESSAGE = (
    "Model nije dostupan na cloudu jer nije postavljen Hugging Face repo ID."
)
HF_IGNORE_NAMES = {
    ".git",
    ".gitattributes",
    "README.md",
    "README",
}


def get_project_root():
    return Path(__file__).resolve().parents[1]


def ensure_model_available(model_key):
    if is_model_available(model_key):
        return True, "Model je dostupan."

    if model_key not in MODEL_REPO_ENV:
        return False, f"Nepoznat model: {model_key}"

    repo_id = _get_repo_id(model_key)
    if not repo_id:
        return False, MISSING_REPO_MESSAGE

    try:
        if model_key == "resnet50":
            download_resnet50()
        elif model_key == "xlm_roberta":
            download_xlm_roberta()
        elif model_key == "layoutlmv3":
            download_layoutlmv3()
        else:
            return False, f"Nepoznat model: {model_key}"
    except Exception as error:
        return False, f"Preuzimanje modela nije uspjelo: {error}"

    if is_model_available(model_key):
        return True, "Model je preuzet i spreman."

    return False, "Model je preuzet, ali nedostaju očekivane datoteke."


def download_resnet50():
    snapshot_dir = _snapshot_download("resnet50")
    source_dir = _find_first_existing_dir(
        [
            snapshot_dir / "models" / "resnet50",
            snapshot_dir / "resnet50",
            snapshot_dir,
        ]
    )
    target_dir = get_project_root() / "models" / "resnet50"
    target_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = _find_file(source_dir, "label_mapping.json")
    if label_mapping:
        shutil.copy2(label_mapping, target_dir / "label_mapping.json")

    weight_file = _find_resnet_weight(source_dir)
    if weight_file:
        shutil.copy2(weight_file, target_dir / "best_model.pth")

    return target_dir


def download_xlm_roberta():
    snapshot_dir = _snapshot_download("xlm_roberta")
    source_dir = _find_first_existing_dir(
        [
            snapshot_dir / "models" / "xlm_roberta",
            snapshot_dir / "xlm_roberta",
            snapshot_dir,
        ]
    )
    target_parent = get_project_root() / "models" / "xlm_roberta"
    target_model_dir = target_parent / "best_model"
    target_parent.mkdir(parents=True, exist_ok=True)

    if (source_dir / "best_model").is_dir():
        _copy_directory_contents(source_dir / "best_model", target_model_dir)
    else:
        model_dir = _find_transformers_model_dir(source_dir)
        if model_dir:
            _copy_directory_contents(model_dir, target_model_dir)

    label_mapping = _find_file(source_dir, "label_mapping.json")
    if label_mapping:
        shutil.copy2(label_mapping, target_parent / "label_mapping.json")
        if not (target_model_dir / "label_mapping.json").exists():
            shutil.copy2(label_mapping, target_model_dir / "label_mapping.json")

    return target_parent


def download_layoutlmv3():
    snapshot_dir = _snapshot_download("layoutlmv3")
    source_dir = _find_first_existing_dir(
        [
            snapshot_dir / "models" / "layoutlmv3" / "best_model",
            snapshot_dir / "layoutlmv3" / "best_model",
            snapshot_dir / "best_model",
            snapshot_dir,
        ]
    )
    model_dir = _find_transformers_model_dir(source_dir) or source_dir
    target_model_dir = get_project_root() / "models" / "layoutlmv3" / "best_model"
    _copy_directory_contents(model_dir, target_model_dir)

    label_mapping = _find_file(source_dir, "label_mapping.json")
    if label_mapping:
        shutil.copy2(label_mapping, target_model_dir / "label_mapping.json")

    parent_label_mapping = _find_file(snapshot_dir, "label_mapping.json")
    if parent_label_mapping:
        parent_dir = get_project_root() / "models" / "layoutlmv3"
        parent_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_label_mapping, parent_dir / "label_mapping.json")

    return target_model_dir


def is_model_available(model_key):
    root = get_project_root()
    if model_key == "resnet50":
        return _is_resnet50_available(root)
    if model_key == "xlm_roberta":
        return _is_xlm_roberta_available(root)
    if model_key == "layoutlmv3":
        return _is_layoutlmv3_available(root)
    return False


def get_model_status():
    return {
        "resnet50": is_model_available("resnet50"),
        "xlm_roberta": is_model_available("xlm_roberta"),
        "layoutlmv3": is_model_available("layoutlmv3"),
    }


def _is_resnet50_available(root):
    model_dir = root / "models" / "resnet50"
    return (
        (model_dir / "label_mapping.json").exists()
        and (model_dir / "best_model.pth").exists()
    )


def _is_xlm_roberta_available(root):
    model_dir = root / "models" / "xlm_roberta" / "best_model"
    parent_dir = root / "models" / "xlm_roberta"
    has_mapping = (parent_dir / "label_mapping.json").exists() or (
        model_dir / "label_mapping.json"
    ).exists()
    return (
        model_dir.exists()
        and (model_dir / "config.json").exists()
        and _has_transformers_weight(model_dir)
        and _has_tokenizer_file(model_dir)
        and has_mapping
    )


def _is_layoutlmv3_available(root):
    model_dir = root / "models" / "layoutlmv3" / "best_model"
    return (
        model_dir.exists()
        and (model_dir / "config.json").exists()
        and _has_transformers_weight(model_dir)
        and _has_tokenizer_file(model_dir)
    )


def _snapshot_download(model_key):
    repo_id = _get_repo_id(model_key)
    if not repo_id:
        raise RuntimeError(MISSING_REPO_MESSAGE)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Nedostaje dependency huggingface_hub. Instalirajte requirements.txt."
        ) from error

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=_get_hf_token(),
        )
    )


def _get_repo_id(model_key):
    return _get_secret_or_env(MODEL_REPO_ENV[model_key])


def _get_hf_token():
    return _get_secret_or_env("HF_TOKEN")


def _get_secret_or_env(name):
    value = os.environ.get(name)
    if value:
        return value

    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        return None

    return None


def _find_first_existing_dir(candidates):
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[-1]


def _find_transformers_model_dir(source_dir):
    source_dir = Path(source_dir)
    if _looks_like_transformers_model(source_dir):
        return source_dir

    for candidate in source_dir.rglob("config.json"):
        model_dir = candidate.parent
        if _looks_like_transformers_model(model_dir):
            return model_dir

    return None


def _looks_like_transformers_model(path):
    return (
        (path / "config.json").exists()
        and _has_transformers_weight(path)
        and _has_tokenizer_file(path)
    )


def _has_transformers_weight(model_dir):
    return (model_dir / "model.safetensors").exists() or (
        model_dir / "pytorch_model.bin"
    ).exists()


def _has_tokenizer_file(model_dir):
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "vocab.json",
    ]
    return any((model_dir / filename).exists() for filename in tokenizer_files)


def _find_file(source_dir, filename):
    source_dir = Path(source_dir)
    direct = source_dir / filename
    if direct.exists():
        return direct

    for candidate in source_dir.rglob(filename):
        if candidate.is_file():
            return candidate

    return None


def _find_resnet_weight(source_dir):
    source_dir = Path(source_dir)
    preferred_names = ["best_model.pth", "model.pth"]
    for filename in preferred_names:
        found = _find_file(source_dir, filename)
        if found:
            return found

    weights = sorted(source_dir.rglob("*.pth"))
    return weights[0] if weights else None


def _copy_directory_contents(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in source_dir.iterdir():
        if item.name in HF_IGNORE_NAMES:
            continue

        destination = target_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        elif item.is_file():
            shutil.copy2(item, destination)
