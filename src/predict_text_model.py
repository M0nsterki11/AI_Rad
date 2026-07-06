import argparse
import json
import sys
import time
from pathlib import Path

import fitz
import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from docx import Document

    DOCX_AVAILABLE = True
except Exception:
    Document = None
    DOCX_AVAILABLE = False

try:
    from .preprocess import (
        MIN_TEXT_CHARS,
        TESSERACT_AVAILABLE,
        clean_text,
        render_pdf_page,
        run_ocr_on_image,
        strip_html,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from preprocess import (  # type: ignore
        MIN_TEXT_CHARS,
        TESSERACT_AVAILABLE,
        clean_text,
        render_pdf_page,
        run_ocr_on_image,
        strip_html,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "xlm_roberta" / "best_model"
LABEL_MAPPING_PATH = PROJECT_ROOT / "models" / "xlm_roberta" / "label_mapping.json"
MAX_LENGTH = 512
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".html", ".htm", ".docx"}


def load_label_mapping_from_model_config(model_dir=MODEL_DIR):
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        id2label = config.get("id2label")
        label2id = config.get("label2id")
        if isinstance(id2label, dict) and isinstance(label2id, dict):
            class_names = [id2label[str(index)] for index in sorted(int(key) for key in id2label)]
            normalized_label2id = {label: int(index) for label, index in label2id.items()}
            return class_names, normalized_label2id

    if LABEL_MAPPING_PATH.exists():
        mapping = json.loads(LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
        class_names = mapping["class_names"]
        label_to_index = {label: int(index) for label, index in mapping["label_to_index"].items()}
        return class_names, label_to_index

    raise FileNotFoundError(
        f"Cannot find label mapping in {config_path} or {LABEL_MAPPING_PATH}"
    )


def load_text_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing trained XLM-RoBERTa model folder: {MODEL_DIR}")

    class_names, label_to_index = load_label_mapping_from_model_config(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    if model.config.num_labels != len(class_names):
        raise ValueError(
            f"Model has {model.config.num_labels} labels, mapping has {len(class_names)} labels."
        )

    model.to(device)
    model.eval()
    return model, tokenizer, class_names, label_to_index, device


def extract_text_from_pdf(path):
    document = fitz.open(str(path))
    try:
        embedded_text = clean_text("\n".join(page.get_text("text") for page in document))
        if len(embedded_text) >= MIN_TEXT_CHARS:
            return embedded_text

        page_texts = []
        for page_index in range(document.page_count):
            image = render_pdf_page(document.load_page(page_index))
            page_text, _ = run_ocr_on_image(image, "unknown", page_index=page_index)
            if page_text:
                page_texts.append(page_text)
        return clean_text("\n".join(page_texts))
    finally:
        document.close()


def extract_text_from_image(path):
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "Tesseract OCR is not available. Install Tesseract or check preprocess.py configuration."
        )

    with Image.open(path) as image:
        text, _ = run_ocr_on_image(image.convert("RGB"), "unknown", page_index=0)
    return clean_text(text)


def extract_text_from_txt(path):
    return clean_text(Path(path).read_text(encoding="utf-8", errors="ignore"))


def extract_text_from_docx(path):
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not installed, so DOCX text cannot be extracted.")

    document = Document(path)
    return clean_text("\n".join(paragraph.text for paragraph in document.paragraphs))


def extract_text_from_file(path):
    path = Path(path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{extension}'. Supported: {supported}")

    if extension == ".pdf":
        text = extract_text_from_pdf(path)
    elif extension in {".png", ".jpg", ".jpeg"}:
        text = extract_text_from_image(path)
    elif extension == ".txt":
        text = extract_text_from_txt(path)
    elif extension in {".html", ".htm"}:
        text = strip_html(Path(path).read_text(encoding="utf-8", errors="ignore"))
    elif extension == ".docx":
        text = extract_text_from_docx(path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

    text = clean_text(text)
    if len(text) < MIN_TEXT_CHARS:
        raise ValueError(
            f"Document does not contain enough readable text "
            f"({len(text)} characters, minimum {MIN_TEXT_CHARS})."
        )
    return text


@torch.no_grad()
def predict_text(text, model=None, tokenizer=None, class_names=None, device=None, max_length=MAX_LENGTH):
    text = clean_text(text)
    if len(text) < MIN_TEXT_CHARS:
        raise ValueError(
            f"Text is too short for prediction ({len(text)} characters, minimum {MIN_TEXT_CHARS})."
        )

    if model is None or tokenizer is None or class_names is None:
        model, tokenizer, class_names, _, device = load_text_model(device)
    elif device is None:
        device = next(model.parameters()).device

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    start = time.perf_counter()
    logits = model(**encoded).logits
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
        "text_length": len(text),
    }


def predict_file(path, model=None, tokenizer=None, class_names=None, device=None, max_length=MAX_LENGTH):
    text = extract_text_from_file(path)
    result = predict_text(
        text,
        model=model,
        tokenizer=tokenizer,
        class_names=class_names,
        device=device,
        max_length=max_length,
    )
    result["file"] = str(Path(path))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Predict document class with trained XLM-RoBERTa.")
    parser.add_argument("--file", required=True, help="Path to PDF, image, TXT, HTML, or DOCX document.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, class_names, label_to_index, device = load_text_model(device)
    result = predict_file(
        args.file,
        model=model,
        tokenizer=tokenizer,
        class_names=class_names,
        device=device,
    )
    result["label_to_index"] = label_to_index
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
