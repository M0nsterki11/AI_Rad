from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
MODEL_DIR = PROJECT_ROOT / "models" / "xlm_roberta_multipage"
RESULTS_DIR = PROJECT_ROOT / "results" / "xlm_roberta_multipage"
MODEL_NAME = "xlm-roberta-base"
CLASS_NAMES = ["invoice", "cv", "contract", "email", "scientific"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
RANDOM_SEED = 42


class DocumentChunkDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = {
            "input_ids": [int(value) for value in row["input_ids"]],
            "attention_mask": [int(value) for value in row["attention_mask"]],
            "label": LABEL_TO_INDEX[str(row["label"])],
            "document_id": str(row["document_id"]),
            "chunk_index": int(row["chunk_index"]),
        }
        if row.get("token_type_ids") is not None:
            item["token_type_ids"] = [int(value) for value in row["token_type_ids"]]
        return item


class ChunkCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        features = [dict(feature) for feature in features]
        document_ids = [feature.pop("document_id") for feature in features]
        chunk_indices = [feature.pop("chunk_index") for feature in features]
        labels = torch.tensor(
            [feature.pop("label") for feature in features], dtype=torch.long
        )
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        batch["document_ids"] = document_ids
        batch["chunk_indices"] = chunk_indices
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-page XLM-RoBERTa classifier.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_chunk_rows(rows_by_split, max_length):
    if max_length != 512:
        raise ValueError(
            "Chunk artifacts were generated with max_length=512. Rebuild them before "
            "training with another --max-length value."
        )
    for split, rows in rows_by_split.items():
        for row in rows:
            input_ids = row.get("input_ids")
            attention_mask = row.get("attention_mask")
            if not isinstance(input_ids, list) or not input_ids:
                raise ValueError(f"Empty input_ids in {split}: {row.get('document_id')}")
            if len(input_ids) != len(attention_mask or []):
                raise ValueError(
                    f"input_ids/attention_mask mismatch: {row.get('document_id')}"
                )
            if len(input_ids) > max_length:
                raise ValueError(
                    f"Chunk exceeds {max_length} tokens: {row.get('document_id')}"
                )


def make_loaders(rows_by_split, tokenizer, batch_size, num_workers):
    collator = ChunkCollator(tokenizer)
    pin_memory = torch.cuda.is_available()
    return {
        "train": DataLoader(
            DocumentChunkDataset(rows_by_split["train"]),
            batch_size=batch_size,
            sampler=make_document_balanced_sampler(rows_by_split["train"]),
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "validation": DataLoader(
            DocumentChunkDataset(rows_by_split["validation"]),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            DocumentChunkDataset(rows_by_split["test"]),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def split_batch(batch, device):
    document_ids = batch.pop("document_ids")
    batch.pop("chunk_indices")
    labels = batch.pop("labels").to(device)
    model_inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    return model_inputs, labels, document_ids


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    item_count = 0
    logits_by_document = defaultdict(list)
    labels_by_document = {}
    for batch in loader:
        model_inputs, labels, document_ids = split_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**model_inputs, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(outputs.loss.item()) * batch_size
        item_count += batch_size
        append_batch_logits(
            logits_by_document, labels_by_document, document_ids, labels, outputs.logits
        )
    metrics, _ = aggregate_evaluation(
        logits_by_document, labels_by_document, class_count=len(CLASS_NAMES)
    )
    metrics["loss"] = total_loss / item_count if item_count else 0.0
    return metrics


@torch.no_grad()
def collect_evaluation(model, loader, device, measure_prediction_time=False):
    model.eval()
    total_loss = 0.0
    item_count = 0
    elapsed = 0.0
    logits_by_document = defaultdict(list)
    labels_by_document = {}
    for batch in loader:
        model_inputs, labels, document_ids = split_batch(batch, device)
        if measure_prediction_time and device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(**model_inputs, labels=labels)
        if measure_prediction_time and device.type == "cuda":
            torch.cuda.synchronize()
        if measure_prediction_time:
            elapsed += time.perf_counter() - start
        batch_size = labels.size(0)
        total_loss += float(outputs.loss.item()) * batch_size
        item_count += batch_size
        append_batch_logits(
            logits_by_document, labels_by_document, document_ids, labels, outputs.logits
        )
    return {
        "loss": total_loss / item_count if item_count else 0.0,
        "items": item_count,
        "elapsed": elapsed,
        "logits_by_document": dict(logits_by_document),
        "labels_by_document": labels_by_document,
    }


def aggregate_raw(raw, *, method=None, select_on_validation=False):
    metrics, comparisons = aggregate_evaluation(
        raw["logits_by_document"],
        raw["labels_by_document"],
        class_count=len(CLASS_NAMES),
        method=method,
        select_on_validation=select_on_validation,
    )
    metrics["loss"] = raw["loss"]
    metrics["chunks_evaluated"] = raw["items"]
    metrics["prediction_time_seconds"] = raw["elapsed"]
    document_count = int(metrics["documents_evaluated"])
    metrics["seconds_per_document"] = raw["elapsed"] / document_count if document_count else 0.0
    return metrics, comparisons


def model_target(smoke_test):
    return MODEL_DIR / ("smoke_test_best_model" if smoke_test else "best_model")


def results_target(smoke_test):
    return RESULTS_DIR / "smoke_test" if smoke_test else RESULTS_DIR


def save_best_model(model, tokenizer, model_dir, metrics, comparisons):
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    mapping = {
        "class_names": CLASS_NAMES,
        "label_to_index": LABEL_TO_INDEX,
        "index_to_label": {str(index): label for index, label in enumerate(CLASS_NAMES)},
    }
    (model_dir / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    save_aggregation_config(
        model_dir / "aggregation_config.json",
        method=str(metrics["aggregation_method"]),
        top_k=int(metrics["aggregation_top_k"]),
        validation_comparisons=comparisons,
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


def save_test_results(results_dir, metrics, raw):
    results_dir.mkdir(parents=True, exist_ok=True)
    ignored = {"y_true", "y_pred", "document_ids", "probabilities", "per_class"}
    serializable = {key: value for key, value in metrics.items() if key not in ignored}
    (results_dir / "test_metrics.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
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
    with (results_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for label, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([label, *row])
    save_confusion_png(results_dir / "confusion_matrix.png", matrix)
    with (results_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["document_id", "true_label", "predicted_label", "confidence", "chunks_analyzed"]
        )
        for document_id, true, pred in zip(metrics["document_ids"], y_true, y_pred):
            probabilities = metrics["probabilities"][document_id]
            writer.writerow(
                [
                    document_id,
                    CLASS_NAMES[true],
                    CLASS_NAMES[pred],
                    probabilities[pred],
                    len(raw["logits_by_document"][document_id]),
                ]
            )


def is_cuda_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
    set_seed(RANDOM_SEED)
    documents, rows_by_split, artifact_path = load_multipage_training_rows(
        "chunk", smoke_test=args.smoke_test
    )
    validate_chunk_rows(rows_by_split, args.max_length)
    print(f"Authoritative manifest validated: {len(documents)} documents")
    print(f"Chunk manifest: {artifact_path}")
    print(f"Document counts: {split_document_counts(rows_by_split)}")
    print(f"Chunk counts: { {name: len(rows) for name, rows in rows_by_split.items()} }")
    if args.preflight_only:
        print("Preflight passed. Training was not started.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        loaders = make_loaders(rows_by_split, tokenizer, args.batch_size, args.num_workers)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(CLASS_NAMES),
            id2label={index: label for index, label in enumerate(CLASS_NAMES)},
            label2id=LABEL_TO_INDEX,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        best_f1 = -1.0
        patience = 2
        stale_epochs = 0
        history = []
        model_dir = model_target(args.smoke_test)
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, loaders["train"], optimizer, device)
            validation_raw = collect_evaluation(model, loaders["validation"], device)
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
                    "validation_macro_precision": validation_metrics["macro_precision"],
                    "validation_macro_recall": validation_metrics["macro_recall"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "aggregation_method": validation_metrics["aggregation_method"],
                }
            )
            print(
                f"Epoch {epoch}/{args.epochs} | train loss {train_metrics['loss']:.4f} | "
                f"train document accuracy {train_metrics['accuracy']:.4f} | "
                f"validation loss {validation_metrics['loss']:.4f} | "
                f"validation accuracy {validation_metrics['accuracy']:.4f} | "
                f"precision {validation_metrics['macro_precision']:.4f} | "
                f"recall {validation_metrics['macro_recall']:.4f} | "
                f"macro F1 {validation_metrics['macro_f1']:.4f} | "
                f"aggregation {validation_metrics['aggregation_method']}"
            )
            if float(validation_metrics["macro_f1"]) > best_f1:
                best_f1 = float(validation_metrics["macro_f1"])
                stale_epochs = 0
                save_best_model(model, tokenizer, model_dir, validation_metrics, comparisons)
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping after {epoch} epochs.")
                    break

        results_dir = results_target(args.smoke_test)
        results_dir.mkdir(parents=True, exist_ok=True)
        with (results_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        best_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        aggregation = json.loads((model_dir / "aggregation_config.json").read_text(encoding="utf-8"))
        test_raw = collect_evaluation(
            best_model, loaders["test"], device, measure_prediction_time=True
        )
        test_metrics, _ = aggregate_raw(test_raw, method=str(aggregation["method"]))
        save_test_results(results_dir, test_metrics, test_raw)
        print(
            f"TEST document accuracy={test_metrics['accuracy']:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f} "
            f"aggregation={test_metrics['aggregation_method']}"
        )
    except RuntimeError as error:
        if is_cuda_oom(error):
            raise RuntimeError(
                "CUDA out of memory. Retry with a smaller --batch-size."
            ) from error
        raise


if __name__ == "__main__":
    main()
