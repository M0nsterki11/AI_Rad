import argparse
import csv
import gc
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from .predict_layoutlm import load_image_and_ocr, load_layoutlm_model, predict_layoutlm
    from .predict_resnet import load_model as load_resnet_model
    from .predict_resnet import predict_file as predict_resnet_file
    from .predict_text_model import extract_text_from_file, load_text_model, predict_text
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from predict_layoutlm import load_image_and_ocr, load_layoutlm_model, predict_layoutlm  # type: ignore
    from predict_resnet import load_model as load_resnet_model  # type: ignore
    from predict_resnet import predict_file as predict_resnet_file  # type: ignore
    from predict_text_model import extract_text_from_file, load_text_model, predict_text  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_RAW_DIR = PROJECT_ROOT / "data" / "external_test" / "raw"
INTERNAL_RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results" / "external_test"
EXCLUDED_DUPLICATES_PATH = RESULTS_DIR / "excluded_duplicates.csv"

CLASS_NAMES = ["invoice", "cv", "contract", "email", "scientific"]
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MODEL_ORDER = [
    ("resnet50", "ResNet50"),
    ("xlm_roberta", "XLM-RoBERTa"),
    ("layoutlmv3", "LayoutLMv3"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all trained models on external test documents.")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional maximum number of external documents to evaluate per class.",
    )
    return parser.parse_args()


def project_relative(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_internal_raw_hash_index():
    hashes = defaultdict(list)
    if not INTERNAL_RAW_DIR.exists():
        return hashes

    for path in sorted(INTERNAL_RAW_DIR.rglob("*")):
        if not path.is_file():
            continue
        try:
            hashes[sha256_file(path)].append(project_relative(path))
        except OSError:
            continue

    return hashes


def collect_external_documents():
    if not EXTERNAL_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing external test folder: {EXTERNAL_RAW_DIR}")

    documents = []
    for label in CLASS_NAMES:
        class_dir = EXTERNAL_RAW_DIR / label
        if not class_dir.exists():
            continue

        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            documents.append(
                {
                    "path": path.resolve(),
                    "document_path": project_relative(path),
                    "true_label": label,
                    "sha256": sha256_file(path),
                }
            )

    return documents


def exclude_internal_duplicates(documents, internal_hashes):
    kept = []
    excluded = []

    for document in documents:
        matches = internal_hashes.get(document["sha256"], [])
        if matches:
            excluded.append(
                {
                    "document_path": document["document_path"],
                    "true_label": document["true_label"],
                    "sha256": document["sha256"],
                    "matching_raw_paths": " | ".join(matches),
                }
            )
        else:
            kept.append(document)

    return kept, excluded


def apply_limit_per_class(documents, limit):
    if limit is None:
        return documents
    if limit < 1:
        raise ValueError("--limit-per-class must be a positive integer.")

    limited = []
    for label in CLASS_NAMES:
        label_docs = [doc for doc in documents if doc["true_label"] == label]
        limited.extend(label_docs[:limit])
    return limited


def write_excluded_duplicates(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with EXCLUDED_DUPLICATES_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["document_path", "true_label", "sha256", "matching_raw_paths"],
        )
        writer.writeheader()
        writer.writerows(rows)


def probability_dict(result):
    probabilities = result.get("probabilities", {})
    if isinstance(probabilities, dict):
        return {label: float(probabilities.get(label, 0.0)) for label in CLASS_NAMES}

    rows = {}
    for item in probabilities:
        label = item.get("class")
        if label in CLASS_NAMES:
            rows[label] = float(item.get("probability", 0.0))
    return {label: rows.get(label, 0.0) for label in CLASS_NAMES}


def success_record(model_key, document, result):
    predicted_label = result.get("predicted_class", "")
    if predicted_label not in CLASS_NAMES:
        raise ValueError(f"Model returned unexpected label: {predicted_label}")

    probabilities = probability_dict(result)
    record = {
        "model": model_key,
        "document_path": document["document_path"],
        "true_label": document["true_label"],
        "predicted_label": predicted_label,
        "confidence": float(result.get("confidence", 0.0)),
        "prediction_time_seconds": float(result.get("prediction_time_seconds", 0.0)),
        "status": "success",
        "error_message": "",
    }
    for label in CLASS_NAMES:
        record[f"prob_{label}"] = probabilities[label]
    return record


def failure_record(model_key, document, error):
    record = {
        "model": model_key,
        "document_path": document["document_path"],
        "true_label": document["true_label"],
        "predicted_label": "",
        "confidence": "",
        "prediction_time_seconds": "",
        "status": "failed",
        "error_message": str(error),
    }
    for label in CLASS_NAMES:
        record[f"prob_{label}"] = ""
    return record


def cleanup_after_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_resnet(documents):
    model_key = "resnet50"
    records = []
    if not documents:
        return records

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    try:
        model, class_names, _ = load_resnet_model(device)
        for document in documents:
            try:
                result = predict_resnet_file(
                    document["path"],
                    model=model,
                    class_names=class_names,
                    device=device,
                )
                records.append(success_record(model_key, document, result))
            except Exception as error:
                records.append(failure_record(model_key, document, error))
    except Exception as error:
        records = [failure_record(model_key, document, error) for document in documents]
    finally:
        del model
        cleanup_after_model()

    return records


def evaluate_xlm_roberta(documents):
    model_key = "xlm_roberta"
    records = []
    if not documents:
        return records

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tokenizer = None
    try:
        model, tokenizer, class_names, _, device = load_text_model(device)
        for document in documents:
            try:
                text = extract_text_from_file(document["path"])
                result = predict_text(
                    text,
                    model=model,
                    tokenizer=tokenizer,
                    class_names=class_names,
                    device=device,
                )
                records.append(success_record(model_key, document, result))
            except Exception as error:
                records.append(failure_record(model_key, document, error))
    except Exception as error:
        records = [failure_record(model_key, document, error) for document in documents]
    finally:
        del model
        del tokenizer
        cleanup_after_model()

    return records


def evaluate_layoutlmv3(documents):
    model_key = "layoutlmv3"
    records = []
    if not documents:
        return records

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    processor = None
    try:
        model, processor, class_names, _, device = load_layoutlm_model(device)
        for document in documents:
            try:
                image, words, boxes, _ = load_image_and_ocr(document["path"])
                result = predict_layoutlm(
                    image,
                    words,
                    boxes,
                    model=model,
                    processor=processor,
                    class_names=class_names,
                    device=device,
                )
                records.append(success_record(model_key, document, result))
            except Exception as error:
                records.append(failure_record(model_key, document, error))
    except Exception as error:
        records = [failure_record(model_key, document, error) for document in documents]
    finally:
        del model
        del processor
        cleanup_after_model()

    return records


def classification_metrics(y_true, y_pred):
    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []

    for label in CLASS_NAMES:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        support = sum(1 for true in y_true if true == label)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true) if y_true else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
    }


def confusion_matrix(y_true, y_pred):
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    label_to_index = {label: index for index, label in enumerate(CLASS_NAMES)}
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            matrix[label_to_index[true]][label_to_index[pred]] += 1
    return matrix


def classification_report_text(metrics):
    lines = ["label,precision,recall,f1,support"]
    for label in CLASS_NAMES:
        row = metrics["per_class"][label]
        lines.append(
            f"{label},{row['precision']:.6f},{row['recall']:.6f},{row['f1']:.6f},{row['support']}"
        )
    lines.append("")
    lines.append(f"accuracy,{metrics['accuracy']:.6f}")
    lines.append(f"macro_precision,{metrics['macro_precision']:.6f}")
    lines.append(f"macro_recall,{metrics['macro_recall']:.6f}")
    lines.append(f"macro_f1,{metrics['macro_f1']:.6f}")
    return "\n".join(lines)


def save_confusion_matrix_csv(path, matrix):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for label, row in zip(CLASS_NAMES, matrix):
            writer.writerow([label, *row])


def save_confusion_matrix_png(path, matrix, title):
    values = [value for row in matrix for value in row]
    max_value = max(values) if values else 1
    cell = 86
    left = 150
    top = 110
    width = left + cell * len(CLASS_NAMES) + 40
    height = top + cell * len(CLASS_NAMES) + 80

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        label_font = ImageFont.truetype("arial.ttf", 16)
        cell_font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        cell_font = ImageFont.load_default()

    def text_center(box, text, font, fill="black"):
        x1, y1, x2, y2 = box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (x1 + (x2 - x1 - text_width) / 2, y1 + (y2 - y1 - text_height) / 2),
            text,
            font=font,
            fill=fill,
        )

    draw.text((left, 25), f"{title} External Test Confusion Matrix", font=title_font, fill="black")
    draw.text((left + cell * len(CLASS_NAMES) / 2 - 50, 70), "Predicted label", font=label_font, fill="black")
    draw.text((20, top + cell * len(CLASS_NAMES) / 2 - 10), "True label", font=label_font, fill="black")

    for column, label in enumerate(CLASS_NAMES):
        text_center((left + column * cell, top - 38, left + (column + 1) * cell, top), label, label_font)

    for row_index, label in enumerate(CLASS_NAMES):
        y1 = top + row_index * cell
        y2 = y1 + cell
        text_center((0, y1, left - 8, y2), label, label_font)

        for column_index, value in enumerate(matrix[row_index]):
            x1 = left + column_index * cell
            x2 = x1 + cell
            intensity = value / max_value if max_value else 0
            shade = int(255 - 170 * intensity)
            color = (shade, shade + int(35 * (1 - intensity)), 255)
            draw.rectangle((x1, y1, x2, y2), fill=color, outline=(80, 100, 130))
            fill = "white" if intensity > 0.65 else "black"
            text_center((x1, y1, x2, y2), str(value), cell_font, fill=fill)

    image.save(path)


def compute_and_save_model_results(model_key, model_name, records):
    model_dir = RESULTS_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    successes = [record for record in records if record["status"] == "success"]
    failures = [record for record in records if record["status"] != "success"]
    y_true = [record["true_label"] for record in successes]
    y_pred = [record["predicted_label"] for record in successes]
    prediction_time = sum(float(record["prediction_time_seconds"]) for record in successes)

    metrics = classification_metrics(y_true, y_pred)
    metrics["prediction_time_seconds"] = prediction_time
    metrics["seconds_per_document"] = prediction_time / len(successes) if successes else 0.0
    metrics["documents_processed"] = len(successes)
    metrics["documents_failed"] = len(failures)
    metrics["documents_total"] = len(records)

    (model_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (model_dir / "classification_report.txt").write_text(
        classification_report_text(metrics) + "\n",
        encoding="utf-8",
    )

    matrix = confusion_matrix(y_true, y_pred)
    save_confusion_matrix_csv(model_dir / "confusion_matrix.csv", matrix)
    save_confusion_matrix_png(model_dir / "confusion_matrix.png", matrix, model_name)

    return metrics


def write_all_predictions(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "document_path",
        "true_label",
        "predicted_label",
        "confidence",
        "prediction_time_seconds",
        "status",
        "error_message",
        *[f"prob_{label}" for label in CLASS_NAMES],
    ]
    with (RESULTS_DIR / "all_predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_comparison_metrics(model_metrics):
    fieldnames = [
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "seconds_per_document",
        "documents_processed",
        "documents_failed",
    ]
    with (RESULTS_DIR / "comparison_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for model_key, model_name in MODEL_ORDER:
            metrics = model_metrics[model_key]
            writer.writerow(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "seconds_per_document": metrics["seconds_per_document"],
                    "documents_processed": metrics["documents_processed"],
                    "documents_failed": metrics["documents_failed"],
                }
            )


def print_comparison_table(model_metrics):
    print()
    print("EXTERNAL TEST COMPARISON")
    print("-" * 105)
    print(
        f"{'Model':<15} {'Accuracy':>10} {'Macro P':>10} {'Macro R':>10} "
        f"{'Macro F1':>10} {'Sec/doc':>10} {'OK':>8} {'Failed':>8}"
    )
    print("-" * 105)
    for model_key, model_name in MODEL_ORDER:
        metrics = model_metrics[model_key]
        print(
            f"{model_name:<15} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['macro_precision']:>10.4f} "
            f"{metrics['macro_recall']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f} "
            f"{metrics['seconds_per_document']:>10.6f} "
            f"{metrics['documents_processed']:>8} "
            f"{metrics['documents_failed']:>8}"
        )
    print("-" * 105)
    print(f"Results saved to: {RESULTS_DIR}")


def print_dataset_summary(documents, excluded):
    counts = defaultdict(int)
    for document in documents:
        counts[document["true_label"]] += 1

    print("External documents selected for evaluation:")
    for label in CLASS_NAMES:
        print(f"  {label}: {counts[label]}")
    print(f"  total: {len(documents)}")
    print(f"Excluded duplicates from data/raw: {len(excluded)}")


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_external_docs = collect_external_documents()
    internal_hashes = build_internal_raw_hash_index()
    documents, excluded = exclude_internal_duplicates(all_external_docs, internal_hashes)
    documents = apply_limit_per_class(documents, args.limit_per_class)
    write_excluded_duplicates(excluded)

    print_dataset_summary(documents, excluded)

    all_records = []
    model_records = {}

    print()
    print("Evaluating ResNet50...")
    model_records["resnet50"] = evaluate_resnet(documents)
    all_records.extend(model_records["resnet50"])

    print("Evaluating XLM-RoBERTa...")
    model_records["xlm_roberta"] = evaluate_xlm_roberta(documents)
    all_records.extend(model_records["xlm_roberta"])

    print("Evaluating LayoutLMv3...")
    model_records["layoutlmv3"] = evaluate_layoutlmv3(documents)
    all_records.extend(model_records["layoutlmv3"])

    write_all_predictions(all_records)

    model_metrics = {}
    for model_key, model_name in MODEL_ORDER:
        model_metrics[model_key] = compute_and_save_model_results(
            model_key,
            model_name,
            model_records[model_key],
        )

    write_comparison_metrics(model_metrics)
    print_comparison_table(model_metrics)


if __name__ == "__main__":
    main()
