import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import fitz
import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoProcessor

try:
    from .preprocess import (
        TESSERACT_AVAILABLE,
        clean_text,
        render_pdf_page,
        run_ocr_on_image,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from preprocess import (  # type: ignore
        TESSERACT_AVAILABLE,
        clean_text,
        render_pdf_page,
        run_ocr_on_image,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "layoutlmv3" / "best_model"
MAX_LENGTH = 512
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "bbox",
    "pixel_values",
    "token_type_ids",
}


def load_label_mapping_from_model_config(model_dir=MODEL_DIR):
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing LayoutLMv3 config file: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    id2label = config.get("id2label")
    label2id = config.get("label2id")
    if not isinstance(id2label, dict) or not isinstance(label2id, dict):
        raise ValueError(f"Model config does not contain id2label/label2id: {config_path}")

    class_names = [id2label[str(index)] for index in sorted(int(key) for key in id2label)]
    normalized_label2id = {label: int(index) for label, index in label2id.items()}
    ordered_from_label2id = [
        label for label, _ in sorted(normalized_label2id.items(), key=lambda item: item[1])
    ]
    if ordered_from_label2id != class_names:
        raise ValueError("LayoutLMv3 id2label and label2id are not consistent.")

    return class_names, normalized_label2id


def load_layoutlm_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing trained LayoutLMv3 model folder: {MODEL_DIR}")

    class_names, label_to_index = load_label_mapping_from_model_config(MODEL_DIR)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, apply_ocr=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    if model.config.num_labels != len(class_names):
        raise ValueError(
            f"Model has {model.config.num_labels} labels, mapping has {len(class_names)} labels."
        )

    model.to(device)
    model.eval()
    return model, processor, class_names, label_to_index, device


def clamp_0_1000(value):
    return max(0, min(1000, int(round(value))))


def normalize_box(raw_box, image_width, image_height):
    if (
        not isinstance(raw_box, list)
        or len(raw_box) != 4
        or not all(isinstance(value, (int, float)) for value in raw_box)
    ):
        raise ValueError(f"Invalid OCR box shape: {raw_box}")

    x1, y1, x2, y2 = [float(value) for value in raw_box]
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid OCR box coordinates: {raw_box}")

    normalized = [
        clamp_0_1000(1000 * x1 / image_width),
        clamp_0_1000(1000 * y1 / image_height),
        clamp_0_1000(1000 * x2 / image_width),
        clamp_0_1000(1000 * y2 / image_height),
    ]

    nx1, ny1, nx2, ny2 = normalized
    if not (0 <= nx1 <= nx2 <= 1000 and 0 <= ny1 <= ny2 <= 1000):
        raise ValueError(f"Invalid normalized OCR box: {normalized}")

    return normalized


def clean_words_and_boxes(words, boxes):
    if len(words) != len(boxes):
        raise ValueError(f"OCR words/boxes mismatch: {len(words)} words, {len(boxes)} boxes.")

    clean_words = []
    clean_boxes = []
    for word, box in zip(words, boxes):
        text = clean_text(word)
        if not text:
            continue
        clean_words.append(text)
        clean_boxes.append(box)

    if not clean_words:
        raise ValueError("OCR did not find any readable words for LayoutLMv3.")

    return clean_words, clean_boxes


def normalize_boxes_for_image(boxes, image):
    image_width, image_height = image.size
    return [normalize_box(box, image_width, image_height) for box in boxes]


def model_input_keys(model):
    forward_keys = set(inspect.signature(model.forward).parameters)
    return MODEL_INPUT_KEYS.intersection(forward_keys)


def render_pdf_first_page(path):
    document = fitz.open(str(path))
    try:
        if document.page_count < 1:
            raise ValueError(f"PDF has no pages: {path}")
        return render_pdf_page(document.load_page(0)).convert("RGB")
    finally:
        document.close()


def load_document_image(path):
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{extension}'. Supported: {supported}")

    if extension == ".pdf":
        return render_pdf_first_page(path)

    with Image.open(path) as image:
        return image.convert("RGB")


def extract_ocr_from_image(image):
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "Tesseract OCR is not available. Install Tesseract or check preprocess.py configuration."
        )

    text, payload = run_ocr_on_image(image.convert("RGB"), "unknown", page_index=0)
    words = payload.get("words", [])
    boxes = payload.get("boxes", [])
    words, boxes = clean_words_and_boxes(words, boxes)
    return words, boxes, clean_text(text)


def load_image_and_ocr(path):
    image = load_document_image(path)
    words, boxes, ocr_text = extract_ocr_from_image(image)
    return image, words, boxes, ocr_text


@torch.no_grad()
def predict_layoutlm(
    image,
    words,
    boxes,
    model=None,
    processor=None,
    class_names=None,
    device=None,
    max_length=MAX_LENGTH,
):
    image = image.convert("RGB")
    words, boxes = clean_words_and_boxes(words, boxes)
    normalized_boxes = normalize_boxes_for_image(boxes, image)

    if model is None or processor is None or class_names is None:
        model, processor, class_names, _, device = load_layoutlm_model(device)
    elif device is None:
        device = next(model.parameters()).device

    encoding = processor(
        image,
        words,
        boxes=normalized_boxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    allowed_keys = model_input_keys(model)
    model_inputs = {
        key: value.to(device)
        for key, value in encoding.items()
        if key in allowed_keys and torch.is_tensor(value)
    }

    start = time.perf_counter()
    logits = model(**model_inputs).logits
    if device.type == "cuda":
        torch.cuda.synchronize()
    prediction_time = time.perf_counter() - start

    probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()
    probabilities = [
        {
            "class": label,
            "probability": float(probabilities_tensor[index].item()),
        }
        for index, label in enumerate(class_names)
    ]
    probabilities.sort(key=lambda item: item["probability"], reverse=True)
    best = probabilities[0]

    return {
        "predicted_class": best["class"],
        "confidence": best["probability"],
        "probabilities": probabilities,
        "prediction_time_seconds": prediction_time,
        "device": str(device),
        "ocr_word_count": len(words),
    }


def predict_file(path, model=None, processor=None, class_names=None, device=None, max_length=MAX_LENGTH):
    image, words, boxes, ocr_text = load_image_and_ocr(path)
    result = predict_layoutlm(
        image,
        words,
        boxes,
        model=model,
        processor=processor,
        class_names=class_names,
        device=device,
        max_length=max_length,
    )
    result["file"] = str(Path(path))
    result["ocr_preview"] = ocr_text[:1000]
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Predict document class with trained LayoutLMv3.")
    parser.add_argument("--file", required=True, help="Path to a PDF, PNG, JPG, or JPEG document.")
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
