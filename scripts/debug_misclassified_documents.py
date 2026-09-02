from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.upload_models_to_hf import DEFAULT_REPOS  # noqa: E402
from src.document_adapter import prepare_document_for_models  # noqa: E402
from src.multipage import (  # noqa: E402
    AGGREGATION_METHODS,
    aggregate_scores,
    normalize_boxes_to_1000,
    tokenize_document_chunks,
)
from src.predict_layoutlm import (  # noqa: E402
    MAX_LENGTH as LAYOUT_MAX_LENGTH,
    active_model_dir as active_layout_model_dir,
    clean_words_and_boxes,
    load_layoutlm_model,
    model_input_keys,
    normalize_boxes_for_image,
    predict_layout_pages,
)
from src.predict_resnet import (  # noqa: E402
    active_model_path as active_resnet_model_path,
    load_label_mapping as load_resnet_label_mapping,
    load_model as load_resnet_model,
    make_preprocess as make_resnet_preprocess,
    predict_images as predict_resnet_images,
)
from src.predict_text_model import (  # noqa: E402
    MAX_LENGTH as XLM_MAX_LENGTH,
    active_model_dir as active_xlm_model_dir,
    load_text_model,
    predict_text,
)
from src.preprocess import MIN_TEXT_CHARS, TESSERACT_AVAILABLE, clean_text  # noqa: E402


CLASS_NAMES = ["invoice", "cv", "contract", "email", "scientific"]
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".docx"}
MODEL_DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "xlm_roberta": "XLM-RoBERTa",
    "layoutlmv3": "LayoutLMv3",
}
MODEL_REPO_ENV = {
    "resnet50": "HF_RESNET50_REPO_ID",
    "xlm_roberta": "HF_XLM_ROBERTA_REPO_ID",
    "layoutlmv3": "HF_LAYOUTLMV3_REPO_ID",
}
REMOTE_WEIGHT_PATHS = {
    "resnet50": "best_model.pth",
    "xlm_roberta": "best_model/model.safetensors",
    "layoutlmv3": "model.safetensors",
}
OUTPUT_SUBDIRS = (
    "extracted_text",
    "xlm_chunks",
    "layout_ocr",
    "rendered_pages",
)

INDICATOR_PATTERNS = {
    "invoice": {
        "racun": r"\b(?:racun|račun)\b",
        "invoice": r"\binvoice\b",
        "OIB": r"\boib\b",
        "PDV": r"\bpdv\b",
        "VAT": r"\bvat\b",
        "IBAN": r"\biban\b",
        "subtotal": r"\bsubtotal\b",
        "ukupno": r"\bukupno\b",
        "total": r"\btotal\b",
        "dospijece": r"\b(?:dospijece|dospijeće)\b",
        "invoice number": r"\binvoice\s+(?:number|no\.?|#)",
        "broj racuna": r"\bbroj\s+(?:racuna|računa)\b",
    },
    "email": {
        "from": r"(?:^|\s)from\s*:",
        "to": r"(?:^|\s)to\s*:",
        "subject": r"(?:^|\s)subject\s*:",
    },
    "cv": {
        "experience": r"\bexperience\b",
        "education": r"\beducation\b",
        "skills": r"\bskills\b",
        "curriculum vitae": r"\bcurriculum\s+vitae\b",
        "resume": r"\bresume\b",
    },
}

FLAT_LABEL_PATTERNS = {
    "invoice": ("invoice", "racun"),
    "cv": ("cv", "resume", "zivotopis"),
    "contract": ("contract", "ugovor"),
    "email": ("email", "mail"),
    "scientific": ("scientific", "paper", "znanstveni"),
}


def parse_args() -> argparse.Namespace:
    default_samples = (
        PROJECT_ROOT / "debug_samples"
        if (PROJECT_ROOT / "debug_samples").is_dir()
        else PROJECT_ROOT / "data" / "debug_samples"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose document misclassifications through the same multi-page "
            "preprocessing and prediction functions used by app.py."
        )
    )
    parser.add_argument("--samples-dir", type=Path, default=default_samples)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "debug_output")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. The default matches the Streamlit application.",
    )
    parser.add_argument(
        "--skip-remote-version-check",
        action="store_true",
        help="Do not query Hugging Face for the latest repo revision and file hash.",
    )
    return parser.parse_args()


def resolve_path(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def safe_slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value[:100] or "document"


def normalized_ascii(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()


def infer_flat_label(filename: str) -> str | None:
    normalized = normalized_ascii(Path(filename).stem)
    for label, indicators in FLAT_LABEL_PATTERNS.items():
        if any(indicator in normalized for indicator in indicators):
            return label
    return None


def collect_documents(samples_dir: Path) -> list[dict[str, Any]]:
    documents = []
    for path in sorted(samples_dir.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(samples_dir)
        expected = next(
            (part.casefold() for part in relative.parts[:-1] if part.casefold() in CLASS_NAMES),
            None,
        )
        label_source = "parent_folder"
        if expected is None:
            expected = infer_flat_label(path.name)
            label_source = "filename_fallback" if expected else "unknown"
        digest = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()[:8]
        documents.append(
            {
                "path": path,
                "relative_path": relative.as_posix(),
                "expected_class": expected,
                "label_source": label_source,
                "slug": f"{safe_slug(path.stem)}_{digest}",
            }
        )
    return documents


def literal_assignment(path: Path, variable_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == variable_name for target in targets):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError):
                return None
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def normalize_id2label(mapping: Mapping[Any, Any] | None) -> dict[str, str]:
    if not isinstance(mapping, Mapping):
        return {}
    normalized = {str(int(key)): str(value) for key, value in mapping.items()}
    return dict(sorted(normalized.items(), key=lambda item: int(item[0])))


def normalize_label2id(mapping: Mapping[Any, Any] | None) -> dict[str, int]:
    if not isinstance(mapping, Mapping):
        return {}
    return {str(label): int(index) for label, index in mapping.items()}


def sibling_lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, Mapping):
        return str(lfs.get("sha256") or "") or None
    return str(getattr(lfs, "sha256", "") or "") or None


def fetch_remote_version(repo_id: str, remote_weight_path: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True, token=False)
        sibling = next(
            (item for item in (info.siblings or []) if item.rfilename == remote_weight_path),
            None,
        )
        return {
            "status": "available",
            "revision": str(info.sha or "") or None,
            "last_modified": json_ready(info.last_modified),
            "remote_weight_path": remote_weight_path,
            "remote_weight_sha256": sibling_lfs_sha256(sibling) if sibling else None,
            "error": "" if sibling else f"Remote weight file is missing: {remote_weight_path}",
        }
    except Exception as error:
        return {
            "status": "unavailable",
            "revision": None,
            "last_modified": None,
            "remote_weight_path": remote_weight_path,
            "remote_weight_sha256": None,
            "error": str(error),
        }


def model_version_record(
    model_key: str,
    local_path: Path,
    weight_path: Path,
    class_names: Sequence[str],
    label_to_index: Mapping[str, int],
    config_id2label: Mapping[Any, Any] | None,
    config_label2id: Mapping[Any, Any] | None,
    training_classes: Sequence[str] | None,
    app_classes: Sequence[str] | None,
    *,
    skip_remote: bool,
) -> dict[str, Any]:
    repo_id = os.environ.get(MODEL_REPO_ENV[model_key], DEFAULT_REPOS[model_key])
    local_sha256 = sha256_file(weight_path)
    remote = (
        {
            "status": "skipped",
            "revision": None,
            "last_modified": None,
            "remote_weight_path": REMOTE_WEIGHT_PATHS[model_key],
            "remote_weight_sha256": None,
            "error": "Remote version check disabled by CLI.",
        }
        if skip_remote
        else fetch_remote_version(repo_id, REMOTE_WEIGHT_PATHS[model_key])
    )
    id2label = normalize_id2label(config_id2label) or {
        str(index): label for index, label in enumerate(class_names)
    }
    normalized_label2id = normalize_label2id(config_label2id) or {
        str(label): int(index) for label, index in label_to_index.items()
    }
    expected_id2label = {str(index): label for index, label in enumerate(class_names)}
    mapping_sources = {
        "predictor_class_names": list(class_names),
        "model_id2label": id2label,
        "model_label2id": normalized_label2id,
        "training_class_names": list(training_classes or []),
        "streamlit_class_names": list(app_classes or []),
    }
    mapping_consistent = (
        list(class_names) == CLASS_NAMES
        and list(training_classes or []) == CLASS_NAMES
        and list(app_classes or []) == CLASS_NAMES
        and id2label == expected_id2label
        and [
            label
            for label, _ in sorted(normalized_label2id.items(), key=lambda item: item[1])
        ]
        == CLASS_NAMES
    )
    remote_sha = remote.get("remote_weight_sha256")
    remote_matches_local = local_sha256 == remote_sha if remote_sha else None
    return {
        "model": MODEL_DISPLAY_NAMES[model_key],
        "repo_id": repo_id,
        "remote_revision": remote.get("revision"),
        "remote_last_modified": remote.get("last_modified"),
        "remote_status": remote.get("status"),
        "remote_error": remote.get("error"),
        "local_model_path": str(local_path),
        "local_weight_path": str(weight_path),
        "local_weight_sha256": local_sha256,
        "remote_weight_path": remote.get("remote_weight_path"),
        "remote_weight_sha256": remote_sha,
        "local_matches_latest_upload": remote_matches_local,
        "id2label": id2label,
        "label2id": normalized_label2id,
        "mapping_sources": mapping_sources,
        "label_mapping_consistent": mapping_consistent,
        "label_mapping_warning": "" if mapping_consistent else "Possible label-mapping mismatch.",
    }


def load_models_and_versions(device: torch.device, skip_remote: bool) -> tuple[dict, dict]:
    print(f"Loading production models on {device}...")
    resnet_model, resnet_classes, resnet_structure = load_resnet_model(device)
    xlm_model, tokenizer, xlm_classes, xlm_label2id, _ = load_text_model(device)
    layout_model, processor, layout_classes, layout_label2id, _ = load_layoutlm_model(device)

    app_classes = literal_assignment(PROJECT_ROOT / "app.py", "CLASS_NAMES")
    training_classes = {
        "resnet50": literal_assignment(PROJECT_ROOT / "src" / "train_resnet.py", "CLASS_NAMES"),
        "xlm_roberta": literal_assignment(
            PROJECT_ROOT / "src" / "train_text_model.py", "CLASS_NAMES"
        ),
        "layoutlmv3": literal_assignment(
            PROJECT_ROOT / "src" / "train_layoutlm.py", "CLASS_NAMES"
        ),
    }

    resnet_path = active_resnet_model_path()
    _, resnet_label2id = load_resnet_label_mapping(resnet_path)
    resnet_checkpoint_id2label = {
        str(index): label
        for index, label in enumerate(
            resnet_structure.get("checkpoint_class_names") or resnet_classes
        )
    }
    resnet_checkpoint_label2id = (
        resnet_structure.get("checkpoint_label_to_index") or resnet_label2id
    )

    model_versions = {
        "resnet50": model_version_record(
            "resnet50",
            resnet_path.parent,
            resnet_path,
            resnet_classes,
            resnet_label2id,
            resnet_checkpoint_id2label,
            resnet_checkpoint_label2id,
            training_classes["resnet50"],
            app_classes,
            skip_remote=skip_remote,
        ),
        "xlm_roberta": model_version_record(
            "xlm_roberta",
            active_xlm_model_dir(),
            active_xlm_model_dir() / "model.safetensors",
            xlm_classes,
            xlm_label2id,
            xlm_model.config.id2label,
            xlm_model.config.label2id,
            training_classes["xlm_roberta"],
            app_classes,
            skip_remote=skip_remote,
        ),
        "layoutlmv3": model_version_record(
            "layoutlmv3",
            active_layout_model_dir(),
            active_layout_model_dir() / "model.safetensors",
            layout_classes,
            layout_label2id,
            layout_model.config.id2label,
            layout_model.config.label2id,
            training_classes["layoutlmv3"],
            app_classes,
            skip_remote=skip_remote,
        ),
    }
    models = {
        "resnet50": (resnet_model, resnet_classes),
        "xlm_roberta": (xlm_model, tokenizer, xlm_classes),
        "layoutlmv3": (layout_model, processor, layout_classes),
    }
    return models, model_versions


def probability_map(result: Mapping[str, Any], class_names: Sequence[str]) -> dict[str, float]:
    probabilities = result.get("probabilities", {})
    if isinstance(probabilities, Mapping):
        return {label: float(probabilities.get(label, 0.0)) for label in class_names}
    rows = {
        str(row.get("class")): float(row.get("probability", 0.0))
        for row in probabilities
        if isinstance(row, Mapping)
    }
    return {label: rows.get(label, 0.0) for label in class_names}


def probabilities_from_tensor(
    probabilities: torch.Tensor, class_names: Sequence[str]
) -> dict[str, float]:
    return {
        label: float(probabilities[index].item())
        for index, label in enumerate(class_names)
    }


def softmax_rows(logits: torch.Tensor, class_names: Sequence[str]) -> list[dict[str, Any]]:
    probabilities = torch.softmax(logits, dim=1)
    rows = []
    for index in range(probabilities.shape[0]):
        probability_map_row = probabilities_from_tensor(probabilities[index], class_names)
        best_index = int(probabilities[index].argmax().item())
        rows.append(
            {
                "predicted_class": class_names[best_index],
                "confidence": probability_map_row[class_names[best_index]],
                "probabilities": probability_map_row,
            }
        )
    return rows


def aggregation_comparison(
    logits: torch.Tensor,
    class_names: Sequence[str],
    production_method: str,
    top_k: int,
) -> dict[str, Any]:
    comparisons = {}
    for method in AGGREGATION_METHODS:
        _, probabilities = aggregate_scores(
            logits,
            method=method,
            top_k=top_k,
            scores_are_logits=True,
        )
        mapped = probabilities_from_tensor(probabilities, class_names)
        best_index = int(probabilities.argmax().item())
        comparisons[method] = {
            "predicted_class": class_names[best_index],
            "confidence": mapped[class_names[best_index]],
            "probabilities": mapped,
            "is_production_method": method == production_method,
        }
    return comparisons


def max_probability_difference(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    labels = set(left) | set(right)
    return max((abs(float(left.get(label, 0.0)) - float(right.get(label, 0.0))) for label in labels), default=0.0)


def selected_page_inputs(prepared: Mapping[str, Any], *, layout: bool = False) -> dict[str, Any]:
    artifact_key = "layout_page_artifacts" if layout else "page_artifacts"
    artifacts = list(prepared.get(artifact_key) or [])
    images = []
    for artifact in artifacts:
        with Image.open(artifact["image_path"]) as source:
            images.append(source.convert("RGB"))
    return {
        "artifacts": artifacts,
        "images": images,
        "words": [list(artifact.get("words", [])) for artifact in artifacts],
        "boxes": [list(artifact.get("boxes", [])) for artifact in artifacts],
        "page_indices": [int(artifact["page_index"]) for artifact in artifacts],
    }


@torch.inference_mode()
def resnet_logits(images: Sequence[Image.Image], model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    preprocess = make_resnet_preprocess()
    tensors = torch.stack([preprocess(image.convert("RGB")) for image in images])
    parts = []
    for start in range(0, len(tensors), 8):
        parts.append(model(tensors[start : start + 8].to(device)).detach().float().cpu())
    return torch.cat(parts, dim=0)


@torch.inference_mode()
def xlm_logits(chunks: Sequence[Mapping[str, Any]], model, tokenizer, device) -> torch.Tensor:
    parts = []
    for start in range(0, len(chunks), 4):
        selected = chunks[start : start + 4]
        features = [
            {
                key: chunk[key]
                for key in ("input_ids", "attention_mask", "token_type_ids")
                if key in chunk
            }
            for chunk in selected
        ]
        encoded = tokenizer.pad(features, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        parts.append(model(**encoded).logits.detach().float().cpu())
    return torch.cat(parts, dim=0)


@torch.inference_mode()
def layout_logits(
    images: Sequence[Image.Image],
    words_by_page: Sequence[Sequence[str]],
    boxes_by_page: Sequence[Sequence[Sequence[int]]],
    model,
    processor,
    device,
) -> tuple[torch.Tensor, list[list[str]], list[list[list[int]]]]:
    rgb_images = []
    cleaned_words = []
    normalized_boxes = []
    for image, words, boxes in zip(images, words_by_page, boxes_by_page):
        rgb = image.convert("RGB")
        words, boxes = clean_words_and_boxes(words, boxes)
        rgb_images.append(rgb)
        cleaned_words.append(words)
        normalized_boxes.append(normalize_boxes_for_image(boxes, rgb))

    allowed_keys = model_input_keys(model)
    parts = []
    for start in range(0, len(rgb_images), 2):
        stop = start + 2
        encoding = processor(
            images=rgb_images[start:stop],
            text=cleaned_words[start:stop],
            boxes=normalized_boxes[start:stop],
            truncation=True,
            padding="max_length",
            max_length=LAYOUT_MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device)
            for key, value in encoding.items()
            if key in allowed_keys and torch.is_tensor(value)
        }
        parts.append(model(**inputs).logits.detach().float().cpu())
    return torch.cat(parts, dim=0), cleaned_words, normalized_boxes


def indicator_hits(text: str) -> dict[str, list[str]]:
    lowered = str(text or "").casefold()
    return {
        group: [name for name, pattern in patterns.items() if re.search(pattern, lowered, re.MULTILINE)]
        for group, patterns in INDICATOR_PATTERNS.items()
    }


def infer_text_source(path: Path, prepared: Mapping[str, Any]) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        import fitz

        try:
            document = fitz.open(str(path))
            try:
                embedded = clean_text("\n".join(page.get_text("text") for page in document))
            finally:
                document.close()
        except Exception:
            embedded = ""
        return "pdf_embedded_text" if len(embedded) >= MIN_TEXT_CHARS else "tesseract_ocr_fallback"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "tesseract_ocr"
    if suffix == ".txt":
        return "direct_txt"
    if suffix == ".docx":
        return "docx_embedded_text"
    return "unknown"


def ocr_languages(prepared: Mapping[str, Any], text_source: str) -> list[str]:
    methods = {
        str(page.get("extraction_method", ""))
        for page in (prepared.get("page_artifacts") or [])
    }
    used = text_source.startswith("tesseract") or any("tesseract" in method for method in methods)
    if not used:
        return []
    return ["eng (Tesseract default; run_ocr_on_image passes no explicit lang argument)"]


def boxes_diagnostics(boxes: Sequence[Sequence[int]], image: Image.Image) -> dict[str, Any]:
    raw_valid = True
    normalized_valid = True
    normalized_boxes = []
    invalid_indices = []
    for index, box in enumerate(boxes):
        try:
            if len(box) != 4:
                raise ValueError("box does not have four coordinates")
            x1, y1, x2, y2 = [int(value) for value in box]
            if x1 < 0 or y1 < 0 or x2 < x1 or y2 < y1 or x2 > image.width or y2 > image.height:
                raise ValueError("box is outside image bounds")
            normalized = normalize_boxes_to_1000([box], image.width, image.height)[0]
            if any(value < 0 or value > 1000 for value in normalized):
                normalized_valid = False
            normalized_boxes.append(normalized)
        except Exception:
            raw_valid = False
            normalized_valid = False
            invalid_indices.append(index)
    return {
        "raw_boxes_within_image": raw_valid,
        "normalized_boxes_within_0_1000": normalized_valid,
        "invalid_box_indices": invalid_indices,
        "normalized_boxes": normalized_boxes,
    }


def copy_rendered_pages(page_inputs: Mapping[str, Any], output_dir: Path, slug: str) -> list[str]:
    document_dir = output_dir / "rendered_pages" / slug
    document_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for image, page_index in zip(page_inputs["images"], page_inputs["page_indices"]):
        target = document_dir / f"page_{int(page_index) + 1:04d}.png"
        image.convert("RGB").save(target, format="PNG")
        saved.append(target.relative_to(output_dir).as_posix())
    return saved


def diagnose_resnet(prepared, models, output_dir: Path, slug: str, device) -> dict[str, Any]:
    model, class_names = models["resnet50"]
    inputs = selected_page_inputs(prepared)
    if not inputs["images"]:
        return {"status": "failed", "error": "No valid page images for ResNet50."}
    saved_images = copy_rendered_pages(inputs, output_dir, slug)
    try:
        production = predict_resnet_images(
            inputs["images"],
            page_indices=inputs["page_indices"],
            total_pages=prepared.get("total_pages"),
            model=model,
            class_names=class_names,
            device=device,
        )
        logits = resnet_logits(inputs["images"], model, device)
        per_page = softmax_rows(logits, class_names)
        for row, page_index, image_path in zip(per_page, inputs["page_indices"], saved_images):
            row.update({"page_index": page_index, "rendered_image": image_path})
        method = str(production["aggregation_method"])
        top_k = int(production["aggregation_top_k"])
        comparisons = aggregation_comparison(logits, class_names, method, top_k)
        production_map = probability_map(production, class_names)
        return {
            "status": "success",
            "error": "",
            "production_result": production,
            "before_aggregation": per_page,
            "after_aggregation": production_map,
            "aggregation_comparison": comparisons,
            "production_replay_max_probability_difference": max_probability_difference(
                production_map, comparisons[method]["probabilities"]
            ),
            "rendered_images": saved_images,
        }
    except Exception as error:
        return {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
    finally:
        for image in inputs["images"]:
            image.close()


def diagnose_xlm(
    prepared,
    models,
    output_dir: Path,
    slug: str,
    device,
) -> dict[str, Any]:
    model, tokenizer, class_names = models["xlm_roberta"]
    text_path = prepared.get("text_path")
    if not text_path or not Path(text_path).is_file():
        return {"status": "failed", "error": "No extracted text for XLM-RoBERTa."}
    text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
    extracted_target = output_dir / "extracted_text" / f"{slug}.txt"
    extracted_target.write_text(text, encoding="utf-8")
    try:
        production = predict_text(
            text,
            model=model,
            tokenizer=tokenizer,
            class_names=class_names,
            device=device,
        )
        selected_chunks = tokenize_document_chunks(
            tokenizer,
            text,
            max_length=XLM_MAX_LENGTH,
            stride=64,
            max_chunks=12,
        )
        all_chunks = tokenize_document_chunks(
            tokenizer,
            text,
            max_length=XLM_MAX_LENGTH,
            stride=64,
            max_chunks=100000,
        )
        logits = xlm_logits(selected_chunks, model, tokenizer, device)
        chunk_rows = softmax_rows(logits, class_names)
        chunk_dir = output_dir / "xlm_chunks" / slug
        chunk_dir.mkdir(parents=True, exist_ok=True)
        selected_text_parts = []
        for row, chunk in zip(chunk_rows, selected_chunks):
            chunk_index = int(chunk["chunk_index"])
            chunk_text = tokenizer.decode(chunk["input_ids"], skip_special_tokens=True)
            chunk_target = chunk_dir / f"chunk_{chunk_index:04d}.txt"
            chunk_target.write_text(chunk_text, encoding="utf-8")
            selected_text_parts.append(chunk_text)
            row.update(
                {
                    "chunk_index": chunk_index,
                    "token_count": len(chunk["input_ids"]),
                    "chunk_file": chunk_target.relative_to(output_dir).as_posix(),
                    "indicator_hits": indicator_hits(chunk_text),
                }
            )

        all_chunk_indicators = {
            int(chunk["chunk_index"]): indicator_hits(
                tokenizer.decode(chunk["input_ids"], skip_special_tokens=True)
            )
            for chunk in all_chunks
        }
        selected_indices = {int(chunk["chunk_index"]) for chunk in selected_chunks}
        omitted_indicator_hits = {
            str(index): hits
            for index, hits in all_chunk_indicators.items()
            if index not in selected_indices and any(hits.values())
        }
        method = str(production["aggregation_method"])
        top_k = int(production["aggregation_top_k"])
        comparisons = aggregation_comparison(logits, class_names, method, top_k)
        production_map = probability_map(production, class_names)
        final_class = str(production["predicted_class"])
        contributors = sorted(
            (
                {
                    "chunk_index": row["chunk_index"],
                    "probability_for_final_class": row["probabilities"][final_class],
                    "predicted_class": row["predicted_class"],
                }
                for row in chunk_rows
            ),
            key=lambda row: row["probability_for_final_class"],
            reverse=True,
        )[:3]
        full_tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        return {
            "status": "success",
            "error": "",
            "production_result": production,
            "text_file": extracted_target.relative_to(output_dir).as_posix(),
            "character_count": len(text),
            "word_count": len(text.split()),
            "token_count": len(full_tokens),
            "total_chunks": len(all_chunks),
            "selected_chunk_indices": sorted(selected_indices),
            "selected_chunks": chunk_rows,
            "before_aggregation": chunk_rows,
            "after_aggregation": production_map,
            "aggregation_comparison": comparisons,
            "top_contributing_chunks": contributors,
            "full_text_indicator_hits": indicator_hits(text),
            "selected_text_indicator_hits": indicator_hits("\n".join(selected_text_parts)),
            "omitted_chunk_indicator_hits": omitted_indicator_hits,
            "production_replay_max_probability_difference": max_probability_difference(
                production_map, comparisons[method]["probabilities"]
            ),
        }
    except Exception as error:
        return {
            "status": "failed",
            "error": str(error),
            "text_file": extracted_target.relative_to(output_dir).as_posix(),
            "character_count": len(text),
            "word_count": len(text.split()),
            "traceback": traceback.format_exc(),
        }


def diagnose_layout(prepared, models, output_dir: Path, slug: str, device) -> dict[str, Any]:
    model, processor, class_names = models["layoutlmv3"]
    inputs = selected_page_inputs(prepared, layout=True)
    if not inputs["images"]:
        return {"status": "failed", "error": "No valid aligned image/OCR pages for LayoutLMv3."}
    try:
        production = predict_layout_pages(
            inputs["images"],
            inputs["words"],
            inputs["boxes"],
            page_indices=inputs["page_indices"],
            total_pages=prepared.get("total_pages"),
            model=model,
            processor=processor,
            class_names=class_names,
            device=device,
        )
        logits, cleaned_words, normalized_boxes = layout_logits(
            inputs["images"],
            inputs["words"],
            inputs["boxes"],
            model,
            processor,
            device,
        )
        page_rows = softmax_rows(logits, class_names)
        ocr_dir = output_dir / "layout_ocr" / slug
        ocr_dir.mkdir(parents=True, exist_ok=True)
        warnings = []
        for position, (row, artifact, image, words, boxes, normalized) in enumerate(
            zip(
                page_rows,
                inputs["artifacts"],
                inputs["images"],
                cleaned_words,
                inputs["boxes"],
                normalized_boxes,
            )
        ):
            box_check = boxes_diagnostics(boxes, image)
            payload_path = Path(artifact["ocr_path"])
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                payload_page_index = int(payload.get("page_index", -1))
                payload_image = Path(str(payload.get("image_path", "")))
                aligned = (
                    payload_page_index == int(artifact["page_index"])
                    and payload.get("words", []) == artifact.get("words", [])
                    and payload.get("boxes", []) == artifact.get("boxes", [])
                    and payload_image.resolve() == Path(artifact["image_path"]).resolve()
                )
            except Exception:
                aligned = False
            page_warning = []
            if len(words) < 5:
                page_warning.append("Very short OCR result (fewer than 5 words).")
            if not box_check["raw_boxes_within_image"]:
                page_warning.append("One or more raw bounding boxes are outside the image.")
            if not box_check["normalized_boxes_within_0_1000"]:
                page_warning.append("One or more normalized boxes are outside 0-1000.")
            if not aligned:
                page_warning.append("OCR words/boxes do not align with the same image/page payload.")
            warnings.extend(
                f"Page {int(artifact['page_index']) + 1}: {message}" for message in page_warning
            )
            saved_payload = {
                "page_index": int(artifact["page_index"]),
                "image_width": image.width,
                "image_height": image.height,
                "extraction_method": artifact.get("extraction_method"),
                "layout_status": artifact.get("layout_status"),
                "words": words,
                "boxes_pixels": boxes,
                "boxes_normalized_0_1000": normalized,
                "box_validation": {key: value for key, value in box_check.items() if key != "normalized_boxes"},
                "aligned_with_image_and_page": aligned,
                "warnings": page_warning,
            }
            target = ocr_dir / f"page_{int(artifact['page_index']) + 1:04d}.json"
            target.write_text(json.dumps(saved_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            row.update(
                {
                    "page_index": int(artifact["page_index"]),
                    "ocr_word_count": len(words),
                    "first_300_ocr_words": " ".join(words[:300]),
                    "extraction_method": artifact.get("extraction_method"),
                    "raw_boxes_within_image": box_check["raw_boxes_within_image"],
                    "normalized_boxes_within_0_1000": box_check[
                        "normalized_boxes_within_0_1000"
                    ],
                    "aligned_with_image_and_page": aligned,
                    "ocr_file": target.relative_to(output_dir).as_posix(),
                    "warnings": page_warning,
                }
            )
        method = str(production["aggregation_method"])
        top_k = int(production["aggregation_top_k"])
        comparisons = aggregation_comparison(logits, class_names, method, top_k)
        production_map = probability_map(production, class_names)
        return {
            "status": "success",
            "error": "",
            "production_result": production,
            "before_aggregation": page_rows,
            "after_aggregation": production_map,
            "aggregation_comparison": comparisons,
            "warnings": warnings,
            "production_replay_max_probability_difference": max_probability_difference(
                production_map, comparisons[method]["probabilities"]
            ),
        }
    except Exception as error:
        return {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
    finally:
        for image in inputs["images"]:
            image.close()


def expected_indicators_missing_from_selection(xlm: Mapping[str, Any], expected: str | None) -> bool:
    if expected not in INDICATOR_PATTERNS:
        return False
    full_hits = (xlm.get("full_text_indicator_hits") or {}).get(expected, [])
    selected_hits = (xlm.get("selected_text_indicator_hits") or {}).get(expected, [])
    return bool(full_hits and not selected_hits)


def any_correct_item(model_result: Mapping[str, Any], expected: str | None) -> bool:
    if not expected:
        return False
    return any(
        row.get("predicted_class") == expected
        for row in (model_result.get("before_aggregation") or [])
    )


def model_is_wrong(model_result: Mapping[str, Any], expected: str | None) -> bool:
    if not expected or model_result.get("status") != "success":
        return False
    production = model_result.get("production_result") or {}
    return production.get("predicted_class") != expected


def annotate_expected_result(
    model_key: str, result: dict[str, Any], expected: str | None
) -> None:
    if not expected or result.get("status") != "success":
        return
    production = result.get("production_result") or {}
    final_prediction = str(production.get("predicted_class", ""))
    correct_items = []
    for row in result.get("before_aggregation") or []:
        if row.get("predicted_class") != expected:
            continue
        item_key = "chunk_index" if model_key == "xlm_roberta" else "page_index"
        correct_items.append(int(row.get(item_key, 0)))
    correct_alternatives = [
        method
        for method, comparison in (result.get("aggregation_comparison") or {}).items()
        if comparison.get("predicted_class") == expected
    ]
    aggregation_overrode_correct_item = final_prediction != expected and bool(correct_items)
    result.update(
        {
            "expected_class": expected,
            "final_top1_correct": final_prediction == expected,
            "correct_item_indices_before_aggregation": correct_items,
            "alternative_aggregations_predicting_expected": correct_alternatives,
            "aggregation_overrode_correct_item": aggregation_overrode_correct_item,
        }
    )
    if aggregation_overrode_correct_item:
        unit = "chunk" if model_key == "xlm_roberta" else "page"
        result.setdefault("diagnostic_warnings", []).append(
            f"AGGREGATION WARNING: at least one {unit} predicted {expected}, "
            f"but final aggregation predicted {final_prediction}."
        )


def choose_conclusion(document: Mapping[str, Any], versions: Mapping[str, Any]) -> tuple[str, str]:
    expected = document.get("expected_class")
    results = document.get("models", {})
    wrong_models = [
        MODEL_DISPLAY_NAMES[key]
        for key, result in results.items()
        if model_is_wrong(result, expected)
    ]
    failed_models = [
        MODEL_DISPLAY_NAMES[key]
        for key, result in results.items()
        if result.get("status") != "success"
    ]
    if not expected:
        return "INPUT/OCR PROBLEM", "Expected class is unknown because no class subfolder was provided."
    if any(info.get("local_matches_latest_upload") is False for info in versions.values()):
        return "MODEL VERSION PROBLEM", "At least one active local weight file differs from the latest Hugging Face upload."
    if any(not info.get("label_mapping_consistent", False) for info in versions.values()):
        return "LABEL MAPPING PROBLEM", "Model, training, and Streamlit label mappings are not identical."
    if not wrong_models and not failed_models:
        return "NO PROBLEM REPRODUCED", "All three production predictions match the expected class."

    basic = document.get("basic", {})
    layout = results.get("layoutlmv3", {})
    severe_ocr = (
        int(basic.get("character_count", 0)) < MIN_TEXT_CHARS
        or not document.get("layout_pages")
        or any(
            not row.get("aligned_with_image_and_page", True)
            or not row.get("raw_boxes_within_image", True)
            or int(row.get("ocr_word_count", 0)) < 5
            for row in (layout.get("before_aggregation") or [])
        )
    )
    if failed_models or severe_ocr:
        evidence = []
        if failed_models:
            evidence.append("technical input failure: " + ", ".join(failed_models))
        if severe_ocr:
            evidence.append("text/OCR is empty, very short, out of bounds, or misaligned")
        return "INPUT/OCR PROBLEM", "; ".join(evidence) + "."

    xlm = results.get("xlm_roberta", {})
    if model_is_wrong(xlm, expected) and expected_indicators_missing_from_selection(xlm, expected):
        return (
            "CHUNK SELECTION PROBLEM",
            "Expected-class indicators exist in the full text but not in the selected XLM chunks.",
        )

    aggregation_models = [
        MODEL_DISPLAY_NAMES[key]
        for key, result in results.items()
        if model_is_wrong(result, expected) and any_correct_item(result, expected)
    ]
    if aggregation_models:
        return (
            "AGGREGATION PROBLEM",
            "At least one page/chunk predicted the expected class, but final aggregation did not: "
            + ", ".join(aggregation_models)
            + ".",
        )

    return (
        "MODEL/DATASET PROBLEM",
        "Input, OCR, version, and mappings passed diagnostics, but these models remain wrong: "
        + ", ".join(wrong_models or failed_models)
        + ".",
    )


def process_document(
    item: Mapping[str, Any],
    models: Mapping[str, Any],
    versions: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(item["path"])
    print(f"Diagnosing: {item['relative_path']}")
    with tempfile.TemporaryDirectory(prefix="document_debug_") as temporary_dir:
        try:
            prepared = prepare_document_for_models(path, Path(temporary_dir))
        except Exception as error:
            document = {
                **{key: value for key, value in item.items() if key != "path"},
                "absolute_path": str(path),
                "preparation_status": "failed",
                "preparation_errors": [str(error)],
                "basic": {},
                "models": {
                    key: {"status": "failed", "error": "Document preparation failed."}
                    for key in MODEL_DISPLAY_NAMES
                },
                "traceback": traceback.format_exc(),
            }
            document["conclusion"], document["conclusion_evidence"] = choose_conclusion(
                document, versions
            )
            return document

        text = ""
        if prepared.get("text_path") and Path(prepared["text_path"]).is_file():
            text = Path(prepared["text_path"]).read_text(encoding="utf-8", errors="ignore")
        text_source = infer_text_source(path, prepared)
        resnet = diagnose_resnet(prepared, models, output_dir, item["slug"], device)
        xlm = diagnose_xlm(prepared, models, output_dir, item["slug"], device)
        layout = diagnose_layout(prepared, models, output_dir, item["slug"], device)
        model_results = {
            "resnet50": resnet,
            "xlm_roberta": xlm,
            "layoutlmv3": layout,
        }
        for model_key, result in model_results.items():
            annotate_expected_result(model_key, result, item.get("expected_class"))

        page_artifacts = prepared.get("page_artifacts") or []
        layout_pages = prepared.get("layout_page_artifacts") or []
        document = {
            **{key: value for key, value in item.items() if key != "path"},
            "absolute_path": str(path),
            "preparation_status": "success",
            "preparation_errors": list(prepared.get("errors") or []),
            "basic": {
                "total_pages": int(prepared.get("total_pages") or 0),
                "selected_page_indices_zero_based": [
                    int(index) for index in (prepared.get("selected_page_indices") or [])
                ],
                "analyzed_page_indices_zero_based": [
                    int(index) for index in (prepared.get("analyzed_page_indices") or [])
                ],
                "analyzed_pages_one_based": [
                    int(index) + 1 for index in (prepared.get("analyzed_page_indices") or [])
                ],
                "character_count": len(text),
                "word_count": len(text.split()),
                "token_count": xlm.get("token_count", 0),
                "xlm_total_chunks": xlm.get("total_chunks", 0),
                "xlm_selected_chunks": len(xlm.get("selected_chunks") or []),
                "text_extraction_source": text_source,
                "ocr_languages": ocr_languages(prepared, text_source),
                "tesseract_available": TESSERACT_AVAILABLE,
            },
            "page_artifacts": [
                {
                    "page_index": int(page.get("page_index", 0)),
                    "image_path": str(page.get("image_path", "")),
                    "ocr_path": str(page.get("ocr_path", "")),
                    "extraction_method": page.get("extraction_method"),
                    "layout_status": page.get("layout_status"),
                    "failure_reason": page.get("failure_reason"),
                    "ocr_word_count": len(page.get("words", [])),
                }
                for page in page_artifacts
            ],
            "layout_pages": [int(page["page_index"]) for page in layout_pages],
            "text_indicator_hits": indicator_hits(text),
            "models": model_results,
        }
        document["conclusion"], document["conclusion_evidence"] = choose_conclusion(
            document, versions
        )
        return document


def result_prediction(result: Mapping[str, Any]) -> tuple[str, float, str]:
    if result.get("status") != "success":
        return "", 0.0, str(result.get("error", "failed"))
    production = result.get("production_result") or {}
    return (
        str(production.get("predicted_class", "")),
        float(production.get("confidence", 0.0)),
        "",
    )


def summary_row(document: Mapping[str, Any]) -> dict[str, Any]:
    expected = str(document.get("expected_class") or "")
    basic = document.get("basic") or {}
    row = {
        "document": document.get("relative_path"),
        "expected_class": expected,
        "expected_label_source": document.get("label_source"),
        "total_pages": basic.get("total_pages", 0),
        "analyzed_page_indices": json.dumps(
            basic.get("analyzed_page_indices_zero_based", []), ensure_ascii=False
        ),
        "characters": basic.get("character_count", 0),
        "words": basic.get("word_count", 0),
        "tokens": basic.get("token_count", 0),
        "xlm_chunks": basic.get("xlm_selected_chunks", 0),
        "text_source": basic.get("text_extraction_source", ""),
        "ocr_languages": "; ".join(basic.get("ocr_languages", [])),
        "conclusion": document.get("conclusion"),
        "evidence": document.get("conclusion_evidence"),
    }
    for key in MODEL_DISPLAY_NAMES:
        prediction, confidence, error = result_prediction((document.get("models") or {}).get(key, {}))
        row[f"{key}_prediction"] = prediction
        row[f"{key}_confidence"] = confidence
        row[f"{key}_correct"] = bool(expected and prediction == expected)
        row[f"{key}_error"] = error
    return row


def write_summary_csv(path: Path, documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [summary_row(document) for document in documents]
    fieldnames = list(rows[0]) if rows else ["document", "expected_class", "conclusion"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def percent(value: Any) -> str:
    try:
        return f"{100 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def format_probability_map(values: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{label}={percent(values.get(label, 0.0))}" for label in CLASS_NAMES
    )


def write_diagnostic_report(
    path: Path,
    documents: Sequence[Mapping[str, Any]],
    versions: Mapping[str, Any],
    overall: Mapping[str, Any],
    samples_dir: Path,
) -> None:
    lines = [
        "DOCUMENT AI CLASSIFIER - MISCLASSIFICATION DIAGNOSTIC",
        "=" * 62,
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Samples: {samples_dir}",
        "Pipeline: prepare_document_for_models + production predict_* functions used by app.py",
        "",
        "MODEL VERSIONS AND LABEL MAPPINGS",
        "-" * 62,
    ]
    for key, info in versions.items():
        match = info.get("local_matches_latest_upload")
        match_text = "YES" if match is True else "NO" if match is False else "UNKNOWN"
        lines.extend(
            [
                MODEL_DISPLAY_NAMES[key],
                f"  Hugging Face repo: {info.get('repo_id')}",
                f"  Remote revision: {info.get('remote_revision') or 'unavailable'}",
                f"  Remote last modified: {info.get('remote_last_modified') or 'unavailable'}",
                f"  Active local path: {info.get('local_model_path')}",
                f"  Local equals latest remote weight: {match_text}",
                f"  id2label: {json.dumps(info.get('id2label'), ensure_ascii=False)}",
                f"  label2id: {json.dumps(info.get('label2id'), ensure_ascii=False)}",
                f"  Mapping consistent across model/training/app: {info.get('label_mapping_consistent')}",
            ]
        )
        if info.get("remote_error"):
            lines.append(f"  Remote check note: {info['remote_error']}")
        if info.get("label_mapping_warning"):
            lines.append(f"  WARNING: {info['label_mapping_warning']}")
        lines.append("")

    for document in documents:
        basic = document.get("basic") or {}
        lines.extend(
            [
                "=" * 62,
                f"DOCUMENT: {document.get('relative_path')}",
                f"Expected class: {document.get('expected_class') or 'UNKNOWN'} "
                f"(source: {document.get('label_source')})",
                f"Pages: {basic.get('total_pages', 0)}; analyzed zero-based: "
                f"{basic.get('analyzed_page_indices_zero_based', [])}",
                f"Text: {basic.get('character_count', 0)} characters, "
                f"{basic.get('word_count', 0)} words, {basic.get('token_count', 0)} tokens",
                f"XLM chunks: {basic.get('xlm_selected_chunks', 0)} selected of "
                f"{basic.get('xlm_total_chunks', 0)}",
                f"Text source: {basic.get('text_extraction_source', 'unknown')}",
                f"OCR languages: {', '.join(basic.get('ocr_languages', [])) or 'not used'}",
                f"Preparation warnings: {document.get('preparation_errors') or 'none'}",
                f"Indicators: {json.dumps(document.get('text_indicator_hits', {}), ensure_ascii=False)}",
                "",
            ]
        )
        for key, result in (document.get("models") or {}).items():
            lines.append(MODEL_DISPLAY_NAMES[key])
            if result.get("status") != "success":
                lines.append(f"  FAILED: {result.get('error')}")
                lines.append("")
                continue
            production = result.get("production_result") or {}
            lines.extend(
                [
                    f"  Final prediction: {production.get('predicted_class')} "
                    f"({percent(production.get('confidence'))})",
                    f"  Production aggregation: {production.get('aggregation_method')} "
                    f"(top_k={production.get('aggregation_top_k')})",
                    f"  Final probabilities: {format_probability_map(result.get('after_aggregation') or {})}",
                    f"  Replay max probability difference: "
                    f"{float(result.get('production_replay_max_probability_difference', 0.0)):.8f}",
                    "  Diagnostic aggregation comparison:",
                ]
            )
            for method, comparison in (result.get("aggregation_comparison") or {}).items():
                marker = " [PRODUCTION]" if comparison.get("is_production_method") else ""
                lines.append(
                    f"    {method}{marker}: {comparison.get('predicted_class')} "
                    f"({percent(comparison.get('confidence'))})"
                )
            if key == "xlm_roberta":
                lines.append("  Selected chunks before aggregation:")
                for row in result.get("selected_chunks") or []:
                    lines.append(
                        f"    chunk {row.get('chunk_index')}: {row.get('predicted_class')} "
                        f"({percent(row.get('confidence'))}), tokens={row.get('token_count')}, "
                        f"probs=[{format_probability_map(row.get('probabilities') or {})}]"
                    )
                lines.append(
                    "  Top contributors to final class: "
                    + json.dumps(result.get("top_contributing_chunks", []), ensure_ascii=False)
                )
                lines.append(
                    "  Omitted chunks with indicators: "
                    + json.dumps(result.get("omitted_chunk_indicator_hits", {}), ensure_ascii=False)
                )
            else:
                lines.append("  Pages before aggregation:")
                for row in result.get("before_aggregation") or []:
                    extra = ""
                    if key == "layoutlmv3":
                        extra = (
                            f", OCR words={row.get('ocr_word_count')}, "
                            f"boxes_valid={row.get('normalized_boxes_within_0_1000')}, "
                            f"aligned={row.get('aligned_with_image_and_page')}"
                        )
                    lines.append(
                        f"    page {int(row.get('page_index', 0)) + 1}: "
                        f"{row.get('predicted_class')} ({percent(row.get('confidence'))}){extra}; "
                        f"probs=[{format_probability_map(row.get('probabilities') or {})}]"
                    )
                    if key == "layoutlmv3":
                        lines.append(
                            "      First 300 OCR words: "
                            + str(row.get("first_300_ocr_words", ""))
                        )
            for warning in result.get("warnings") or []:
                lines.append(f"  WARNING: {warning}")
            for warning in result.get("diagnostic_warnings") or []:
                lines.append(f"  {warning}")
            lines.append("")

        lines.extend(
            [
                f"CONCLUSION: {document.get('conclusion')}",
                f"Evidence: {document.get('conclusion_evidence')}",
                "",
            ]
        )

    lines.extend(
        [
            "=" * 62,
            "JOINT SUMMARY",
            f"Documents analyzed: {overall.get('documents_analyzed')}",
            f"INPUT/OCR PROBLEM: {overall.get('input_ocr_problems', 0)}",
            f"CHUNK SELECTION PROBLEM: {overall.get('chunk_selection_problems', 0)}",
            f"AGGREGATION PROBLEM: {overall.get('aggregation_problems', 0)}",
            f"LABEL MAPPING PROBLEM: {overall.get('label_mapping_problems', 0)}",
            f"MODEL VERSION PROBLEM: {overall.get('model_version_problems', 0)}",
            f"Likely needs new training data (MODEL/DATASET PROBLEM): "
            f"{overall.get('model_dataset_problems', 0)}",
            f"No problem reproduced: {overall.get('no_problem_reproduced', 0)}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def overall_summary(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(document.get("conclusion", "")) for document in documents)
    return {
        "documents_analyzed": len(documents),
        "input_ocr_problems": counts["INPUT/OCR PROBLEM"],
        "chunk_selection_problems": counts["CHUNK SELECTION PROBLEM"],
        "aggregation_problems": counts["AGGREGATION PROBLEM"],
        "label_mapping_problems": counts["LABEL MAPPING PROBLEM"],
        "model_version_problems": counts["MODEL VERSION PROBLEM"],
        "model_dataset_problems": counts["MODEL/DATASET PROBLEM"],
        "no_problem_reproduced": counts["NO PROBLEM REPRODUCED"],
    }


def choose_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    samples_dir = resolve_path(args.samples_dir)
    output_dir = resolve_path(args.output_dir)
    if not samples_dir.is_dir():
        raise FileNotFoundError(
            f"Debug samples directory does not exist: {samples_dir}. "
            "Create class subfolders such as debug_samples/invoice."
        )
    documents_to_process = collect_documents(samples_dir)
    if not documents_to_process:
        raise ValueError(f"No supported documents found under: {samples_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_SUBDIRS:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    fallback_documents = [
        item["relative_path"]
        for item in documents_to_process
        if item["label_source"] == "filename_fallback"
    ]
    if fallback_documents:
        print(
            "WARNING: These samples are not in class subfolders; expected labels were inferred "
            "only for diagnostic ground truth and are never passed to a model: "
            + ", ".join(fallback_documents)
        )

    device = choose_device(args.device)
    models, versions = load_models_and_versions(device, args.skip_remote_version_check)
    reports = [
        process_document(item, models, versions, output_dir, device)
        for item in documents_to_process
    ]
    overall = overall_summary(reports)
    summary_rows = write_summary_csv(output_dir / "summary.csv", reports)
    full_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_dir": str(samples_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "production_pipeline": {
            "document_preparation": "src.document_adapter.prepare_document_for_models",
            "resnet_prediction": "src.predict_resnet.predict_images",
            "xlm_prediction": "src.predict_text_model.predict_text",
            "layout_prediction": "src.predict_layoutlm.predict_layout_pages",
            "page_selection": "src.multipage.select_representative_pages",
            "chunking": "src.multipage.tokenize_document_chunks",
            "aggregation": "src.multipage.aggregate_scores",
        },
        "model_versions": versions,
        "overall_summary": overall,
        "documents": reports,
    }
    (output_dir / "full_report.json").write_text(
        json.dumps(json_ready(full_report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_diagnostic_report(
        output_dir / "diagnostic_report.txt", reports, versions, overall, samples_dir
    )

    print("\nDiagnostic summary")
    print("=" * 80)
    for row in summary_rows:
        print(
            f"{row['document']}: expected={row['expected_class'] or 'UNKNOWN'}, "
            f"ResNet50={row['resnet50_prediction'] or 'FAIL'}, "
            f"XLM={row['xlm_roberta_prediction'] or 'FAIL'}, "
            f"LayoutLMv3={row['layoutlmv3_prediction'] or 'FAIL'} -> {row['conclusion']}"
        )
    print("-" * 80)
    print(json.dumps(overall, indent=2))
    print(f"summary.csv: {output_dir / 'summary.csv'}")
    print(f"full_report.json: {output_dir / 'full_report.json'}")
    print(f"diagnostic_report.txt: {output_dir / 'diagnostic_report.txt'}")


if __name__ == "__main__":
    main()
