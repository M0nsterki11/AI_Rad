import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "metadata.csv"
SPLITS_DIR = DATA_DIR / "splits"
MODEL_DIR = PROJECT_ROOT / "models" / "resnet50"
RESULTS_DIR = PROJECT_ROOT / "results" / "resnet50"

CLASS_NAMES = ["invoice", "cv", "contract", "email", "scientific"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
RANDOM_SEED = 42


class DocumentImageDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = resolve_project_path(row["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = LABEL_TO_INDEX[row["label"]]
        return image, torch.tensor(label, dtype=torch.long), row["id"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet50 document classifier.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
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


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required_columns = {"id", "label", "image_path"}
    missing_columns = required_columns.difference(fieldnames)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"metadata.csv is missing required columns: {missing}")

    filtered_rows = []
    missing_images = []

    for row in rows:
        label = str(row.get("label", "")).strip()
        image_path = str(row.get("image_path", "")).strip()

        if label not in LABEL_TO_INDEX:
            continue

        if not image_path:
            missing_images.append((row.get("id", ""), "(empty image_path)"))
            continue

        resolved = resolve_project_path(image_path)
        if not resolved.exists():
            missing_images.append((row.get("id", ""), str(resolved)))
            continue

        row["label"] = label
        row["image_path"] = image_path
        filtered_rows.append(row)

    if missing_images:
        examples = "\n".join(f"  {row_id}: {path}" for row_id, path in missing_images[:20])
        raise FileNotFoundError(
            f"Found {len(missing_images)} metadata rows with missing images. Examples:\n{examples}"
        )

    counts = Counter(row["label"] for row in filtered_rows)
    for label in CLASS_NAMES:
        if counts[label] == 0:
            raise ValueError(f"No rows found for class: {label}")

    return filtered_rows, fieldnames


def limit_for_smoke_test(rows, max_per_class=10):
    rng = random.Random(RANDOM_SEED)
    limited = []

    for label in CLASS_NAMES:
        label_rows = [row for row in rows if row["label"] == label]
        rng.shuffle(label_rows)
        limited.extend(label_rows[:max_per_class])

    rng.shuffle(limited)
    return limited


def stratified_split(rows):
    rng = random.Random(RANDOM_SEED)
    train_rows = []
    val_rows = []
    test_rows = []

    for label in CLASS_NAMES:
        label_rows = [row for row in rows if row["label"] == label]
        rng.shuffle(label_rows)

        if len(label_rows) < 3:
            raise ValueError(f"Class {label} has fewer than 3 rows, cannot split.")

        train_count = int(len(label_rows) * 0.70)
        val_count = int(len(label_rows) * 0.15)

        train_count = max(1, train_count)
        val_count = max(1, val_count)

        if train_count + val_count >= len(label_rows):
            val_count = 1
            train_count = len(label_rows) - 2

        train_rows.extend(label_rows[:train_count])
        val_rows.extend(label_rows[train_count:train_count + val_count])
        test_rows.extend(label_rows[train_count + val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_split(name, rows, fieldnames):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{name}.csv"

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def make_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def make_loaders(train_rows, val_rows, test_rows, batch_size, num_workers):
    transform = make_transforms()
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        DocumentImageDataset(train_rows, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        DocumentImageDataset(val_rows, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        DocumentImageDataset(test_rows, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def make_model(device):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))
    model.to(device)
    return model


def accuracy_from_counts(correct, total):
    return correct / total if total else 0.0


def classification_metrics(y_true, y_pred):
    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []

    for index, label in INDEX_TO_LABEL.items():
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


def confusion_matrix(y_true, y_pred):
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return running_loss / total, accuracy_from_counts(correct, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device, measure_prediction_time=False):
    model.eval()
    running_loss = 0.0
    total = 0
    all_labels = []
    all_predictions = []
    all_ids = []
    prediction_time = 0.0

    for images, labels, row_ids in loader:
        images = images.to(device)
        labels = labels.to(device)

        if measure_prediction_time:
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            prediction_time += time.perf_counter() - start
        else:
            logits = model(images)

        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_ids.extend(row_ids)

    metrics = classification_metrics(all_labels, all_predictions)
    metrics["loss"] = running_loss / total if total else 0.0
    metrics["prediction_time_seconds"] = prediction_time
    metrics["seconds_per_sample"] = prediction_time / total if total else 0.0
    metrics["y_true"] = all_labels
    metrics["y_pred"] = all_predictions
    metrics["ids"] = all_ids
    return metrics


def save_best_model(model, path, epoch, val_metrics):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "label_to_index": LABEL_TO_INDEX,
            "validation_macro_f1": val_metrics["macro_f1"],
        },
        path,
    )


def classification_report_text(metrics):
    lines = []
    lines.append("label,precision,recall,f1,support")
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


def save_results(test_metrics):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    serializable_metrics = {
        key: value
        for key, value in test_metrics.items()
        if key not in {"y_true", "y_pred", "ids"}
    }
    (RESULTS_DIR / "test_metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2),
        encoding="utf-8",
    )

    (RESULTS_DIR / "classification_report.txt").write_text(
        classification_report_text(test_metrics) + "\n",
        encoding="utf-8",
    )

    matrix = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"])
    with (RESULTS_DIR / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for label, row in zip(CLASS_NAMES, matrix):
            writer.writerow([label, *row])

    with (RESULTS_DIR / "test_predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "true_label", "predicted_label"])
        for row_id, true, pred in zip(test_metrics["ids"], test_metrics["y_true"], test_metrics["y_pred"]):
            writer.writerow([row_id, INDEX_TO_LABEL[true], INDEX_TO_LABEL[pred]])


def print_split_summary(name, rows):
    counts = Counter(row["label"] for row in rows)
    details = ", ".join(f"{label}: {counts[label]}" for label in CLASS_NAMES)
    print(f"{name}: {len(rows)} ({details})")


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    if args.smoke_test:
        args.epochs = 1
        print("Smoke test enabled: using up to 10 images per class and 1 epoch.")

    rows, fieldnames = load_metadata()
    if args.smoke_test:
        rows = limit_for_smoke_test(rows)

    train_rows, val_rows, test_rows = stratified_split(rows)
    write_split("train", train_rows, fieldnames)
    write_split("validation", val_rows, fieldnames)
    write_split("test", test_rows, fieldnames)

    print_split_summary("Train", train_rows)
    print_split_summary("Validation", val_rows)
    print_split_summary("Test", test_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = make_loaders(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = make_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    best_val_f1 = -1.0
    best_model_path = MODEL_DIR / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            "Epoch "
            f"{epoch}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"validation loss: {val_metrics['loss']:.4f} | "
            f"validation accuracy: {val_metrics['accuracy']:.4f} | "
            f"validation macro F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            save_best_model(model, best_model_path, epoch, val_metrics)
            print(f"Saved best model: {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device, measure_prediction_time=True)
    save_results(test_metrics)

    print("TEST RESULTS")
    print(f"accuracy: {test_metrics['accuracy']:.4f}")
    print(f"macro precision: {test_metrics['macro_precision']:.4f}")
    print(f"macro recall: {test_metrics['macro_recall']:.4f}")
    print(f"macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"prediction time seconds: {test_metrics['prediction_time_seconds']:.4f}")
    print(f"seconds per sample: {test_metrics['seconds_per_sample']:.6f}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
