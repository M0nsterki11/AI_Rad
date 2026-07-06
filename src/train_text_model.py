import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:
    raise SystemExit(
        "Missing required library torch. Install project requirements before running this script."
    ) from error

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as error:
    raise SystemExit(
        "Missing required library transformers. Install it with: "
        "python -m pip install transformers"
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODEL_DIR = PROJECT_ROOT / "models" / "xlm_roberta"
RESULTS_DIR = PROJECT_ROOT / "results" / "xlm_roberta"
MODEL_NAME = "xlm-roberta-base"
RANDOM_SEED = 42
MIN_SHORT_TEXT_CHARS = 20

DEFAULT_CLASS_NAMES = ["invoice", "cv", "contract", "email", "scientific"]
RESNET_LABEL_MAPPING_PATH = PROJECT_ROOT / "models" / "resnet50" / "label_mapping.json"
LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = row["label_id"]
        encoded["id"] = row["id"]
        return encoded


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        row_ids = [feature.pop("id") for feature in features]
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["ids"] = row_ids
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train XLM-RoBERTa document text classifier.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_project_path(value):
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_label_mapping():
    if RESNET_LABEL_MAPPING_PATH.exists():
        mapping = json.loads(RESNET_LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
        class_names = mapping.get("class_names") or DEFAULT_CLASS_NAMES
        label_to_index = mapping.get("label_to_index") or {
            label: index for index, label in enumerate(class_names)
        }
    else:
        class_names = DEFAULT_CLASS_NAMES
        label_to_index = {label: index for index, label in enumerate(class_names)}

    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(f"Unexpected class order from label mapping: {class_names}")

    expected_mapping = {label: index for index, label in enumerate(DEFAULT_CLASS_NAMES)}
    if label_to_index != expected_mapping:
        raise ValueError(f"Unexpected label_to_index mapping: {label_to_index}")

    index_to_label = {str(index): label for label, index in label_to_index.items()}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_MAPPING_PATH.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "label_to_index": label_to_index,
                "index_to_label": index_to_label,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return class_names, label_to_index


def read_split(split_name):
    path = SPLITS_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split CSV file: {path}. Run/create ResNet splits first."
        )

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = {"id", "label", "text_path"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    return rows


def load_text(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def prepare_rows(rows, split_name, label_to_index):
    prepared = []
    invalid_labels = []
    missing_texts = []
    empty_texts = []
    short_texts = []

    for row in rows:
        row_id = str(row.get("id", "")).strip()
        label = str(row.get("label", "")).strip()
        text_path_value = str(row.get("text_path", "")).strip()
        text_path = resolve_project_path(text_path_value)

        if label not in label_to_index:
            invalid_labels.append({"split": split_name, "id": row_id, "label": label})
            continue

        if not text_path.exists():
            missing_texts.append(
                {
                    "split": split_name,
                    "id": row_id,
                    "label": label,
                    "text_path": str(text_path),
                }
            )
            continue

        text = load_text(text_path)
        clean_text = " ".join(text.split())
        text_length = len(clean_text)

        if text_length == 0:
            empty_texts.append(
                {
                    "split": split_name,
                    "id": row_id,
                    "label": label,
                    "text_path": str(text_path),
                    "text_length": text_length,
                }
            )
        elif text_length < MIN_SHORT_TEXT_CHARS:
            short_texts.append(
                {
                    "split": split_name,
                    "id": row_id,
                    "label": label,
                    "text_path": str(text_path),
                    "text_length": text_length,
                }
            )

        prepared.append(
            {
                "id": row_id,
                "label": label,
                "label_id": label_to_index[label],
                "text": text,
                "text_path": text_path_value,
                "text_length": text_length,
            }
        )

    return prepared, {
        "invalid_labels": invalid_labels,
        "missing_texts": missing_texts,
        "empty_texts": empty_texts,
        "short_texts": short_texts,
    }


def write_issue_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_and_prepare_splits(label_to_index):
    all_prepared = {}
    all_issues = {
        "invalid_labels": [],
        "missing_texts": [],
        "empty_texts": [],
        "short_texts": [],
    }

    for split_name in ["train", "validation", "test"]:
        raw_rows = read_split(split_name)
        prepared_rows, issues = prepare_rows(raw_rows, split_name, label_to_index)
        all_prepared[split_name] = prepared_rows

        for key, value in issues.items():
            all_issues[key].extend(value)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_issue_csv(
        RESULTS_DIR / "empty_text_files.csv",
        all_issues["empty_texts"],
        ["split", "id", "label", "text_path", "text_length"],
    )

    print("TEXT DATA CHECK:")
    for split_name, rows in all_prepared.items():
        counts = Counter(row["label"] for row in rows)
        count_text = ", ".join(f"{label}: {counts[label]}" for label in DEFAULT_CLASS_NAMES)
        print(f"{split_name}: {len(rows)} ({count_text})")
    print(f"empty text files: {len(all_issues['empty_texts'])}")
    print(f"very short text files (<{MIN_SHORT_TEXT_CHARS} chars): {len(all_issues['short_texts'])}")
    print(f"invalid labels: {len(all_issues['invalid_labels'])}")
    print(f"missing text files: {len(all_issues['missing_texts'])}")

    if all_issues["empty_texts"]:
        print(f"WARNING: Empty text files saved to {RESULTS_DIR / 'empty_text_files.csv'}")

    if all_issues["invalid_labels"]:
        examples = all_issues["invalid_labels"][:10]
        raise ValueError(f"Unexpected labels found in split files. Examples: {examples}")

    if all_issues["missing_texts"]:
        examples = all_issues["missing_texts"][:10]
        raise FileNotFoundError(f"Missing text files found. Examples: {examples}")

    return all_prepared


def limit_for_smoke_test(rows_by_split, max_per_class=10):
    rng = random.Random(RANDOM_SEED)
    limited = {}

    for split_name, rows in rows_by_split.items():
        split_rows = []
        for label in DEFAULT_CLASS_NAMES:
            label_rows = [row for row in rows if row["label"] == label]
            rng.shuffle(label_rows)
            split_rows.extend(label_rows[:max_per_class])
        rng.shuffle(split_rows)
        limited[split_name] = split_rows

    return limited


def make_loaders(rows_by_split, tokenizer, batch_size, max_length):
    collator = DataCollator(tokenizer)
    return {
        split_name: DataLoader(
            TextDataset(rows, tokenizer, max_length),
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            collate_fn=collator,
        )
        for split_name, rows in rows_by_split.items()
    }


def classification_metrics(y_true, y_pred, class_names):
    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []

    for index, label in enumerate(class_names):
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == index and pred == index)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != index and pred == index)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == index and pred != index)
        support = sum(1 for true in y_true if true == index)

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


def confusion_matrix(y_true, y_pred, class_names):
    matrix = [[0 for _ in class_names] for _ in class_names]
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if key == "ids":
            moved[key] = value
        else:
            moved[key] = value.to(device)
    return moved


def is_cuda_oom(error):
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def raise_cuda_oom_hint(error):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raise RuntimeError(
        "CUDA out-of-memory during XLM-RoBERTa training/evaluation. "
        "Try a smaller --batch-size, for example --batch-size 1."
    ) from error


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        labels = batch["labels"]
        predictions = outputs.logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

    return {
        "loss": total_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
    }


@torch.no_grad()
def evaluate(model, loader, device, class_names, measure_prediction_time=False):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []
    ids = []
    prediction_time = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]

        start = time.perf_counter()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        if measure_prediction_time and device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        if measure_prediction_time:
            prediction_time += end - start

        predictions = outputs.logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += outputs.loss.item() * batch_size
        total += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        ids.extend(batch["ids"])

    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = total_loss / total if total else 0.0
    metrics["prediction_time_seconds"] = prediction_time
    metrics["seconds_per_document"] = prediction_time / total if total else 0.0
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["ids"] = ids
    return metrics


def classification_report_text(metrics, class_names):
    lines = ["label,precision,recall,f1,support"]
    for label in class_names:
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


def save_confusion_matrix_csv(path, matrix, class_names):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *class_names])
        for label, row in zip(class_names, matrix):
            writer.writerow([label, *row])


def save_confusion_matrix_png(path, matrix, class_names):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("Pillow is required to save confusion_matrix.png") from error

    values = [value for row in matrix for value in row]
    max_value = max(values) if values else 1
    cell = 86
    left = 150
    top = 110
    width = left + cell * len(class_names) + 40
    height = top + cell * len(class_names) + 80

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

    draw.text((left, 25), "XLM-RoBERTa Confusion Matrix", font=title_font, fill="black")
    draw.text((left + cell * len(class_names) / 2 - 50, 70), "Predicted label", font=label_font, fill="black")
    draw.text((20, top + cell * len(class_names) / 2 - 10), "True label", font=label_font, fill="black")

    for column, label in enumerate(class_names):
        text_center((left + column * cell, top - 38, left + (column + 1) * cell, top), label, label_font)

    for row_index, label in enumerate(class_names):
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


def save_training_history(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
                "validation_macro_precision",
                "validation_macro_recall",
                "validation_macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_test_results(results_dir, metrics, class_names):
    results_dir.mkdir(parents=True, exist_ok=True)

    serializable_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"y_true", "y_pred", "ids"}
    }
    (results_dir / "test_metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (results_dir / "classification_report.txt").write_text(
        classification_report_text(metrics, class_names) + "\n",
        encoding="utf-8",
    )

    matrix = confusion_matrix(metrics["y_true"], metrics["y_pred"], class_names)
    save_confusion_matrix_csv(results_dir / "confusion_matrix.csv", matrix, class_names)
    save_confusion_matrix_png(results_dir / "confusion_matrix.png", matrix, class_names)

    with (results_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "true_label", "predicted_label"])
        for row_id, true, pred in zip(metrics["ids"], metrics["y_true"], metrics["y_pred"]):
            writer.writerow([row_id, class_names[true], class_names[pred]])


def target_dirs(smoke_test):
    if smoke_test:
        return MODEL_DIR / "smoke_test_best_model", RESULTS_DIR / "smoke_test"
    return MODEL_DIR / "best_model", RESULTS_DIR


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    if args.smoke_test:
        args.epochs = 1
        print("Smoke test enabled: using up to 10 documents per class per split and 1 epoch.")

    class_names, label_to_index = load_label_mapping()
    rows_by_split = validate_and_prepare_splits(label_to_index)

    if args.smoke_test:
        rows_by_split = limit_for_smoke_test(rows_by_split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(class_names),
            id2label={index: label for index, label in enumerate(class_names)},
            label2id=label_to_index,
        )
    except OSError as error:
        raise RuntimeError(
            f"Could not load pretrained model/tokenizer '{MODEL_NAME}'. "
            "Check internet access or local Hugging Face cache."
        ) from error

    try:
        model.to(device)
    except RuntimeError as error:
        if is_cuda_oom(error):
            raise_cuda_oom_hint(error)
        raise

    loaders = make_loaders(rows_by_split, tokenizer, args.batch_size, args.max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_model_dir, results_dir = target_dirs(args.smoke_test)
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    training_history = []
    patience = 2

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, loaders["train"], optimizer, device)
            val_metrics = evaluate(model, loaders["validation"], device, class_names)

            history_row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_precision": val_metrics["macro_precision"],
                "validation_macro_recall": val_metrics["macro_recall"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
            training_history.append(history_row)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss: {train_metrics['loss']:.4f} | "
                f"train accuracy: {train_metrics['accuracy']:.4f} | "
                f"validation loss: {val_metrics['loss']:.4f} | "
                f"validation accuracy: {val_metrics['accuracy']:.4f} | "
                f"validation macro precision: {val_metrics['macro_precision']:.4f} | "
                f"validation macro recall: {val_metrics['macro_recall']:.4f} | "
                f"validation macro F1: {val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                epochs_without_improvement = 0
                best_model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                print(f"Saved best model to: {best_model_dir}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement.")
                    break

    except RuntimeError as error:
        if is_cuda_oom(error):
            raise_cuda_oom_hint(error)
        raise

    results_dir.mkdir(parents=True, exist_ok=True)
    save_training_history(results_dir / "training_history.csv", training_history)

    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    try:
        model.to(device)
        test_metrics = evaluate(
            model,
            loaders["test"],
            device,
            class_names,
            measure_prediction_time=True,
        )
    except RuntimeError as error:
        if is_cuda_oom(error):
            raise_cuda_oom_hint(error)
        raise
    save_test_results(results_dir, test_metrics, class_names)

    print("TEST RESULTS")
    print(f"accuracy: {test_metrics['accuracy']:.4f}")
    print(f"macro precision: {test_metrics['macro_precision']:.4f}")
    print(f"macro recall: {test_metrics['macro_recall']:.4f}")
    print(f"macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"prediction time seconds: {test_metrics['prediction_time_seconds']:.4f}")
    print(f"seconds per document: {test_metrics['seconds_per_document']:.6f}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
