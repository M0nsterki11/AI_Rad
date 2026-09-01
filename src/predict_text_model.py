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
    from .multipage import aggregate_scores, tokenize_document_chunks
    from .multipage_training import load_aggregation_config
    from .preprocess import (
        MIN_TEXT_CHARS,
        OCR_EMPTY_TEXT_MESSAGE,
        OCRProcessingError,
        TESSERACT_AVAILABLE,
        TESSERACT_UNAVAILABLE_MESSAGE,
        clean_text,
        render_pdf_page,
        run_ocr_on_image,
        strip_html,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from multipage import aggregate_scores, tokenize_document_chunks  # type: ignore
    from multipage_training import load_aggregation_config  # type: ignore
    from preprocess import (  # type: ignore
        MIN_TEXT_CHARS,
        OCR_EMPTY_TEXT_MESSAGE,
        OCRProcessingError,
        TESSERACT_AVAILABLE,
        TESSERACT_UNAVAILABLE_MESSAGE,
        clean_text,
        render_pdf_page,
        run_ocr_on_image,
        strip_html,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTIPAGE_MODEL_DIR = PROJECT_ROOT / "models" / "xlm_roberta_multipage" / "best_model"
LEGACY_MODEL_DIR = PROJECT_ROOT / "models" / "xlm_roberta" / "best_model"
MODEL_DIR = LEGACY_MODEL_DIR
MAX_LENGTH = 512
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".html", ".htm", ".docx"}


def active_model_dir():
    return MULTIPAGE_MODEL_DIR if (MULTIPAGE_MODEL_DIR / "config.json").is_file() else LEGACY_MODEL_DIR


def load_label_mapping_from_model_config(model_dir=None):
    model_dir = Path(model_dir or active_model_dir())
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        id2label = config.get("id2label")
        label2id = config.get("label2id")
        if isinstance(id2label, dict) and isinstance(label2id, dict):
            class_names = [id2label[str(index)] for index in sorted(int(key) for key in id2label)]
            normalized_label2id = {label: int(index) for label, index in label2id.items()}
            return class_names, normalized_label2id

    mapping_paths = [
        Path(model_dir) / "label_mapping.json",
        Path(model_dir).parent / "label_mapping.json",
    ]
    mapping_path = next((path for path in mapping_paths if path.is_file()), None)
    if mapping_path:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        class_names = mapping["class_names"]
        label_to_index = {label: int(index) for label, index in mapping["label_to_index"].items()}
        return class_names, label_to_index

    raise FileNotFoundError(
        f"Cannot find label mapping in {config_path} or beside the selected model."
    )


def load_text_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = active_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing trained XLM-RoBERTa model folder: {model_dir}")

    class_names, label_to_index = load_label_mapping_from_model_config(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    if model.config.num_labels != len(class_names):
        raise ValueError(
            f"Model has {model.config.num_labels} labels, mapping has {len(class_names)} labels."
        )

    model.to(device)
    model.eval()
    method, top_k = load_aggregation_config(model_dir / "aggregation_config.json")
    model.document_aggregation_method = method
    model.document_aggregation_top_k = top_k
    model.document_model_dir = str(model_dir)
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
        ocr_text = clean_text("\n".join(page_texts))
        if len(ocr_text) < MIN_TEXT_CHARS:
            raise OCRProcessingError(OCR_EMPTY_TEXT_MESSAGE)
        return ocr_text
    finally:
        document.close()


def extract_text_from_image(path):
    if not TESSERACT_AVAILABLE:
        raise OCRProcessingError(TESSERACT_UNAVAILABLE_MESSAGE)

    with Image.open(path) as image:
        text, _ = run_ocr_on_image(image.convert("RGB"), "unknown", page_index=0)
    text = clean_text(text)
    if len(text) < MIN_TEXT_CHARS:
        raise OCRProcessingError(OCR_EMPTY_TEXT_MESSAGE)
    return text


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
def predict_text(
    text,
    model=None,
    tokenizer=None,
    class_names=None,
    device=None,
    max_length=MAX_LENGTH,
    batch_size=4,
    aggregation_method=None,
    aggregation_top_k=None,
):
    text = clean_text(text)
    if len(text) < MIN_TEXT_CHARS:
        raise ValueError(
            f"Text is too short for prediction ({len(text)} characters, minimum {MIN_TEXT_CHARS})."
        )

    if model is None or tokenizer is None or class_names is None:
        model, tokenizer, class_names, _, device = load_text_model(device)
    elif device is None:
        device = next(model.parameters()).device

    chunks = tokenize_document_chunks(
        tokenizer,
        text,
        max_length=max_length,
        stride=64,
        max_chunks=12,
    )
    if not chunks:
        raise ValueError("Tokenizer did not produce any chunks for this document.")
    logits_parts = []
    prediction_time = 0.0
    for start_index in range(0, len(chunks), batch_size):
        selected = chunks[start_index : start_index + batch_size]
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(**encoded).logits
        if device.type == "cuda":
            torch.cuda.synchronize()
        prediction_time += time.perf_counter() - start
        logits_parts.append(logits.detach().float().cpu())

    all_logits = torch.cat(logits_parts, dim=0)
    method = aggregation_method or getattr(model, "document_aggregation_method", "top_k_mean")
    top_k = int(aggregation_top_k or getattr(model, "document_aggregation_top_k", 3))
    _, probabilities_tensor = aggregate_scores(
        all_logits, method=method, top_k=top_k, scores_are_logits=True
    )
    probabilities = [
        {
            "class": label,
            "probability": float(probabilities_tensor[index].item()),
        }
        for index, label in enumerate(class_names)
    ]
    probabilities.sort(key=lambda item: item["probability"], reverse=True)
    best = probabilities[0]

    chunk_softmax = torch.softmax(all_logits, dim=1)
    chunk_predictions = []
    for position, chunk in enumerate(chunks):
        best_index = int(chunk_softmax[position].argmax().item())
        chunk_predictions.append(
            {
                "chunk_index": int(chunk["chunk_index"]),
                "predicted_class": class_names[best_index],
                "confidence": float(chunk_softmax[position, best_index]),
                "token_count": len(chunk["input_ids"]),
            }
        )

    return {
        "predicted_class": best["class"],
        "confidence": best["probability"],
        "probabilities": probabilities,
        "prediction_time_seconds": prediction_time,
        "device": str(device),
        "text_length": len(text),
        "total_chunks": int(chunks[0]["total_chunks"]),
        "chunks_analyzed": len(chunks),
        "analyzed_chunk_indices": [int(chunk["chunk_index"]) for chunk in chunks],
        "chunk_predictions": chunk_predictions,
        "aggregation_method": method,
        "aggregation_top_k": top_k,
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
