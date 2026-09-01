from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

try:
    from .multipage_training import (
        aggregate_evaluation,
        append_batch_logits,
        load_multipage_training_rows,
        make_document_balanced_sampler,
        save_aggregation_config,
        split_document_counts,
    )
except ImportError:
    from multipage_training import (  # type: ignore
        aggregate_evaluation,
        append_batch_logits,
        load_multipage_training_rows,
        make_document_balanced_sampler,
        save_aggregation_config,
        split_document_counts,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "resnet50_multipage"
RESULTS_DIR = PROJECT_ROOT / "results" / "resnet50_multipage"
CLASS_NAMES = ["invoice", "cv", "contract", "email", "scientific"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
RANDOM_SEED = 42


class DocumentPageDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = resolve_project_path(row["image_path"])
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return (
            tensor,
            torch.tensor(LABEL_TO_INDEX[str(row["label"])], dtype=torch.long),
            str(row["document_id"]),
            int(row["page_index"]),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-page ResNet50 classifier.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_project_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def make_loaders(rows_by_split, batch_size, num_workers):
    transform = make_transforms()
    pin_memory = torch.cuda.is_available()
    sampler = make_document_balanced_sampler(rows_by_split["train"])
    return {
        "train": DataLoader(
            DocumentPageDataset(rows_by_split["train"], transform),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "validation": DataLoader(
            DocumentPageDataset(rows_by_split["validation"], transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            DocumentPageDataset(rows_by_split["test"], transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def make_model(device):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.to(device)
    return model


def raw_epoch_result(total_loss, item_count, logits_by_document, labels_by_document, elapsed=0.0):
    return {
        "loss": total_loss / item_count if item_count else 0.0,
        "items": item_count,
        "logits_by_document": dict(logits_by_document),
        "labels_by_document": labels_by_document,
        "prediction_time_seconds": elapsed,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    item_count = 0
    logits_by_document = defaultdict(list)
    labels_by_document = {}
    for images, labels, document_ids, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        item_count += batch_size
        append_batch_logits(
            logits_by_document, labels_by_document, document_ids, labels, logits
        )
    raw = raw_epoch_result(total_loss, item_count, logits_by_document, labels_by_document)
    metrics, _ = aggregate_evaluation(
        raw["logits_by_document"], raw["labels_by_document"], class_count=len(CLASS_NAMES)
    )
    metrics["loss"] = raw["loss"]
    return metrics


@torch.no_grad()
def collect_evaluation(model, loader, criterion, device, measure_prediction_time=False):
    model.eval()
    total_loss = 0.0
    item_count = 0
    elapsed = 0.0
    logits_by_document = defaultdict(list)
    labels_by_document = {}
    for images, labels, document_ids, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        start = time.perf_counter()
        logits = model(images)
        if measure_prediction_time and device.type == "cuda":
            torch.cuda.synchronize()
        if measure_prediction_time:
            elapsed += time.perf_counter() - start
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        item_count += batch_size
        append_batch_logits(
            logits_by_document, labels_by_document, document_ids, labels, logits
        )
    return raw_epoch_result(
        total_loss, item_count, logits_by_document, labels_by_document, elapsed
    )


def aggregate_raw(raw, *, method=None, select_on_validation=False):
    metrics, comparisons = aggregate_evaluation(
        raw["logits_by_document"],
        raw["labels_by_document"],
        class_count=len(CLASS_NAMES),
        method=method,
        select_on_validation=select_on_validation,
    )
    metrics["loss"] = raw["loss"]
    metrics["prediction_time_seconds"] = raw["prediction_time_seconds"]
    documents = int(metrics["documents_evaluated"])
    metrics["seconds_per_document"] = (
        raw["prediction_time_seconds"] / documents if documents else 0.0
    )
    metrics["pages_evaluated"] = raw["items"]
    return metrics, comparisons


def save_checkpoint(model, path, epoch, metrics, comparisons):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "label_to_index": LABEL_TO_INDEX,
            "validation_macro_f1": metrics["macro_f1"],
            "aggregation_method": metrics["aggregation_method"],
            "aggregation_top_k": metrics["aggregation_top_k"],
        },
        path,
    )
    save_aggregation_config(
        path.parent / "aggregation_config.json",
        method=str(metrics["aggregation_method"]),
        top_k=int(metrics["aggregation_top_k"]),
        validation_comparisons=comparisons,
    )
    (path.parent / "label_mapping.json").write_text(
        json.dumps({"class_names": CLASS_NAMES, "label_to_index": LABEL_TO_INDEX}, indent=2),
        encoding="utf-8",
    )


def save_confusion_png(path, matrix):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_results(results_dir, metrics):
    results_dir.mkdir(parents=True, exist_ok=True)
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    serializable = {
        key: value
        for key, value in metrics.items()
        if key
        not in {
            "y_true",
            "y_pred",
            "document_ids",
            "probabilities",
            "per_class",
            "page_logits_by_document",
        }
    }
    (results_dir / "test_metrics.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=6,
        zero_division=0,
    )
    (results_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    with (results_dir / "confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for label, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([label, *row])
    save_confusion_png(results_dir / "confusion_matrix.png", matrix)
    with (results_dir / "test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["document_id", "true_label", "predicted_label", "confidence", "pages_analyzed"]
        )
        for document_id, true, pred in zip(metrics["document_ids"], y_true, y_pred):
            probabilities = metrics["probabilities"][document_id]
            writer.writerow(
                [
                    document_id,
                    CLASS_NAMES[true],
                    CLASS_NAMES[pred],
                    probabilities[pred],
                    len(metrics["page_logits_by_document"][document_id]),
                ]
            )


def target_paths(smoke_test):
    if smoke_test:
        return MODEL_DIR / "smoke_test_best_model.pth", RESULTS_DIR / "smoke_test"
    return MODEL_DIR / "best_model.pth", RESULTS_DIR


def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
    set_seed(RANDOM_SEED)
    documents, rows_by_split, artifact_path = load_multipage_training_rows(
        "resnet_page", smoke_test=args.smoke_test
    )
    print(f"Authoritative manifest validated: {len(documents)} documents")
    print(f"Page manifest: {artifact_path}")
    print(f"Document counts: {split_document_counts(rows_by_split)}")
    print(f"Page counts: { {name: len(rows) for name, rows in rows_by_split.items()} }")
    if args.preflight_only:
        print("Preflight passed. Training was not started.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(rows_by_split, args.batch_size, args.num_workers)
    model = make_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    best_path, results_dir = target_paths(args.smoke_test)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        validation_raw = collect_evaluation(model, loaders["validation"], criterion, device)
        validation_metrics, comparisons = aggregate_raw(
            validation_raw, select_on_validation=True
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "aggregation_method": validation_metrics["aggregation_method"],
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs} | train loss {train_metrics['loss']:.4f} | "
            f"train document accuracy {train_metrics['accuracy']:.4f} | "
            f"validation loss {validation_metrics['loss']:.4f} | "
            f"validation document accuracy {validation_metrics['accuracy']:.4f} | "
            f"validation macro F1 {validation_metrics['macro_f1']:.4f} | "
            f"aggregation {validation_metrics['aggregation_method']}"
        )
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = float(validation_metrics["macro_f1"])
            save_checkpoint(model, best_path, epoch, validation_metrics, comparisons)

    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_raw = collect_evaluation(
        model, loaders["test"], criterion, device, measure_prediction_time=True
    )
    test_metrics, _ = aggregate_raw(test_raw, method=checkpoint["aggregation_method"])
    test_metrics["page_logits_by_document"] = test_raw["logits_by_document"]
    save_results(results_dir, test_metrics)
    print(
        f"TEST document accuracy={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"aggregation={test_metrics['aggregation_method']}"
    )


if __name__ == "__main__":
    main()
