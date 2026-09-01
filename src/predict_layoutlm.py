from __future__ import annotations

import argparse
import inspect
import json
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoProcessor

try:
    from .multipage import aggregate_scores, normalize_boxes_to_1000
    from .multipage_preprocess import prepare_inference_file_artifacts
    from .multipage_training import load_aggregation_config
    from .preprocess import (
        OCR_EMPTY_TEXT_MESSAGE,
        OCRProcessingError,
        clean_text,
    )
except ImportError:
    from multipage import aggregate_scores, normalize_boxes_to_1000  # type: ignore
    from multipage_preprocess import prepare_inference_file_artifacts  # type: ignore
    from multipage_training import load_aggregation_config  # type: ignore
    from preprocess import (  # type: ignore
        OCR_EMPTY_TEXT_MESSAGE,
        OCRProcessingError,
        clean_text,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTIPAGE_MODEL_DIR = PROJECT_ROOT / "models" / "layoutlmv3_multipage" / "best_model"
LEGACY_MODEL_DIR = PROJECT_ROOT / "models" / "layoutlmv3" / "best_model"
MODEL_DIR = LEGACY_MODEL_DIR
MAX_LENGTH = 512
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "bbox",
    "pixel_values",
    "token_type_ids",
}


def active_model_dir():
    return MULTIPAGE_MODEL_DIR if (MULTIPAGE_MODEL_DIR / "config.json").is_file() else LEGACY_MODEL_DIR


def load_label_mapping_from_model_config(model_dir=None):
    model_dir = Path(model_dir or active_model_dir())
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing LayoutLMv3 config file: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    id2label = config.get("id2label")
    label2id = config.get("label2id")
    if not isinstance(id2label, dict) or not isinstance(label2id, dict):
        raise ValueError(f"Model config does not contain id2label/label2id: {config_path}")
    class_names = [id2label[str(index)] for index in sorted(int(key) for key in id2label)]
    normalized = {label: int(index) for label, index in label2id.items()}
    ordered = [label for label, _ in sorted(normalized.items(), key=lambda item: item[1])]
    if ordered != class_names:
        raise ValueError("LayoutLMv3 id2label and label2id are inconsistent.")
    return class_names, normalized


def load_layoutlm_model(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = active_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing trained LayoutLMv3 model folder: {model_dir}")
    class_names, label_to_index = load_label_mapping_from_model_config(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    if model.config.num_labels != len(class_names):
        raise ValueError("Model label count does not match its label mapping.")
    model.to(device).eval()
    method, top_k = load_aggregation_config(model_dir / "aggregation_config.json")
    model.document_aggregation_method = method
    model.document_aggregation_top_k = top_k
    model.document_model_dir = str(model_dir)
    return model, processor, class_names, label_to_index, device


def clean_words_and_boxes(words, boxes):
    if len(words) != len(boxes):
        raise ValueError(f"OCR words/boxes mismatch: {len(words)} != {len(boxes)}")
    clean_words = []
    clean_boxes = []
    for word, box in zip(words, boxes):
        text = clean_text(word)
        if text:
            clean_words.append(text)
            clean_boxes.append(box)
    if not clean_words:
        raise ValueError("OCR did not find any readable words for LayoutLMv3.")
    return clean_words, clean_boxes


def normalize_boxes_for_image(boxes, image):
    return normalize_boxes_to_1000(boxes, image.width, image.height)


def model_input_keys(model):
    return MODEL_INPUT_KEYS.intersection(inspect.signature(model.forward).parameters)


@torch.no_grad()
def predict_layout_pages(
    images,
    words_by_page,
    boxes_by_page,
    *,
    page_indices=None,
    total_pages=None,
    model=None,
    processor=None,
    class_names=None,
    device=None,
    max_length=MAX_LENGTH,
    batch_size=2,
    aggregation_method=None,
    aggregation_top_k=None,
):
    if not images or len(images) != len(words_by_page) or len(images) != len(boxes_by_page):
        raise ValueError("LayoutLMv3 requires aligned images, words, and boxes for each page.")
    if model is None or processor is None or class_names is None:
        model, processor, class_names, _, device = load_layoutlm_model(device)
    elif device is None:
        device = next(model.parameters()).device
    page_indices = list(page_indices or range(len(images)))
    if len(page_indices) != len(images):
        raise ValueError("page_indices length does not match page inputs.")

    cleaned_words = []
    normalized_boxes = []
    rgb_images = []
    for image, words, boxes in zip(images, words_by_page, boxes_by_page):
        rgb = image.convert("RGB")
        words, boxes = clean_words_and_boxes(words, boxes)
        rgb_images.append(rgb)
        cleaned_words.append(words)
        normalized_boxes.append(normalize_boxes_for_image(boxes, rgb))

    allowed_keys = model_input_keys(model)
    logits_parts = []
    elapsed = 0.0
    for start_index in range(0, len(rgb_images), batch_size):
        stop = start_index + batch_size
        encoding = processor(
            images=rgb_images[start_index:stop],
            text=cleaned_words[start_index:stop],
            boxes=normalized_boxes[start_index:stop],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device)
            for key, value in encoding.items()
            if key in allowed_keys and torch.is_tensor(value)
        }
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(**inputs).logits
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        logits_parts.append(logits.detach().float().cpu())

    all_logits = torch.cat(logits_parts, dim=0)
    method = aggregation_method or getattr(model, "document_aggregation_method", "top_k_mean")
    top_k = int(aggregation_top_k or getattr(model, "document_aggregation_top_k", 3))
    _, document_probs = aggregate_scores(
        all_logits, method=method, top_k=top_k, scores_are_logits=True
    )
    probability_rows = [
        {"class": label, "probability": float(document_probs[index])}
        for index, label in enumerate(class_names)
    ]
    probability_rows.sort(key=lambda row: row["probability"], reverse=True)
    page_probs = torch.softmax(all_logits, dim=1)
    page_predictions = []
    for position, page_index in enumerate(page_indices):
        best_index = int(page_probs[position].argmax().item())
        page_predictions.append(
            {
                "page_index": int(page_index),
                "predicted_class": class_names[best_index],
                "confidence": float(page_probs[position, best_index]),
                "ocr_word_count": len(cleaned_words[position]),
            }
        )
    best = probability_rows[0]
    return {
        "predicted_class": best["class"],
        "confidence": best["probability"],
        "probabilities": probability_rows,
        "prediction_time_seconds": elapsed,
        "device": str(device),
        "ocr_word_count": sum(len(words) for words in cleaned_words),
        "total_pages": int(total_pages or len(images)),
        "pages_analyzed": len(images),
        "analyzed_page_indices": page_indices,
        "page_predictions": page_predictions,
        "aggregation_method": method,
        "aggregation_top_k": top_k,
    }


def predict_file(path, model=None, processor=None, class_names=None, device=None, max_length=MAX_LENGTH):
    path = Path(path)
    if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported extension '{path.suffix}'. Supported: {supported}")
    with tempfile.TemporaryDirectory(prefix="layout_multipage_") as temporary_dir:
        total_pages, selected_indices, artifacts = prepare_inference_file_artifacts(
            path, Path(temporary_dir)
        )
        artifacts = [artifact for artifact in artifacts if artifact.is_layout_valid]
        if not artifacts:
            raise OCRProcessingError(OCR_EMPTY_TEXT_MESSAGE)
        images = []
        for artifact in artifacts:
            with Image.open(artifact.image_path) as source:
                images.append(source.convert("RGB"))
        result = predict_layout_pages(
            images,
            [artifact.words for artifact in artifacts],
            [artifact.boxes for artifact in artifacts],
            page_indices=[artifact.page_index for artifact in artifacts],
            total_pages=total_pages,
            model=model,
            processor=processor,
            class_names=class_names,
            device=device,
            max_length=max_length,
        )
        result["ocr_preview"] = "\n".join(" ".join(artifact.words) for artifact in artifacts)[:1000]
    result["file"] = str(path)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with multi-page LayoutLMv3.")
    parser.add_argument("--file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, class_names, label_to_index, device = load_layoutlm_model(device)
    result = predict_file(
        args.file,
        model=model,
        processor=processor,
        class_names=class_names,
        device=device,
    )
    result["label_to_index"] = label_to_index
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
