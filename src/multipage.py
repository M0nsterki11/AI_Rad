from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


MAX_SELECTED_PAGES = 12
PAGE_HEAD_COUNT = 4
PAGE_TAIL_COUNT = 3
MAX_SELECTED_CHUNKS = 12
CHUNK_MAX_LENGTH = 512
CHUNK_STRIDE = 64
DEFAULT_AGGREGATION_METHOD = "top_k_mean"
DEFAULT_AGGREGATION_TOP_K = 3
AGGREGATION_METHODS = ("mean", "max", "top_k_mean")
VALID_SPLITS = ("train", "validation", "test")


class ManifestLeakageError(RuntimeError):
    """Raised when related documents occur in more than one split."""


def _evenly_spaced_indices(start: int, end: int, count: int) -> list[int]:
    if count <= 0 or end < start:
        return []
    available = end - start + 1
    if available <= count:
        return list(range(start, end + 1))
    if count == 1:
        return [(start + end) // 2]
    return [round(start + index * (end - start) / (count - 1)) for index in range(count)]


def select_representative_indices(
    total_count: int,
    *,
    maximum: int = MAX_SELECTED_PAGES,
    head_count: int = PAGE_HEAD_COUNT,
    tail_count: int = PAGE_TAIL_COUNT,
) -> list[int]:
    """Select all short inputs or head, evenly spaced middle, and tail indices."""
    if total_count < 0:
        raise ValueError("total_count cannot be negative")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if head_count < 0 or tail_count < 0 or head_count + tail_count > maximum:
        raise ValueError("head_count and tail_count do not fit within maximum")
    if total_count <= maximum:
        return list(range(total_count))

    middle_count = maximum - head_count - tail_count
    head = list(range(head_count))
    tail = list(range(total_count - tail_count, total_count)) if tail_count else []
    middle_start = head_count
    middle_end = total_count - tail_count - 1
    middle = _evenly_spaced_indices(middle_start, middle_end, middle_count)
    selected = sorted(set(head + middle + tail))

    if len(selected) < maximum:
        remaining = [index for index in range(total_count) if index not in selected]
        needed = maximum - len(selected)
        remaining_positions = _evenly_spaced_indices(0, len(remaining) - 1, needed)
        selected.extend(remaining[position] for position in remaining_positions)
        selected = sorted(set(selected))
    return selected[:maximum]


def normalize_box_to_1000(
    raw_box: Sequence[int | float], image_width: int, image_height: int
) -> list[int]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise ValueError(f"Invalid bounding box: {raw_box!r}")
    if not all(isinstance(value, (int, float)) for value in raw_box):
        raise ValueError(f"Bounding box must contain numeric coordinates: {raw_box!r}")
    x1, y1, x2, y2 = [float(value) for value in raw_box]
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid bounding box coordinate order: {raw_box!r}")

    def scaled(value: float, dimension: int) -> int:
        return max(0, min(1000, int(round(1000 * value / dimension))))

    normalized = [
        scaled(x1, image_width),
        scaled(y1, image_height),
        scaled(x2, image_width),
        scaled(y2, image_height),
    ]
    if normalized[2] < normalized[0] or normalized[3] < normalized[1]:
        raise ValueError(f"Invalid normalized bounding box: {normalized!r}")
    return normalized


def normalize_boxes_to_1000(
    boxes: Sequence[Sequence[int | float]], image_width: int, image_height: int
) -> list[list[int]]:
    return [normalize_box_to_1000(box, image_width, image_height) for box in boxes]


def select_representative_pages(total_pages: int) -> list[int]:
    return select_representative_indices(total_pages, maximum=MAX_SELECTED_PAGES)


def select_representative_chunks(total_chunks: int) -> list[int]:
    return select_representative_indices(total_chunks, maximum=MAX_SELECTED_CHUNKS)


def tokenize_document_chunks(
    tokenizer,
    text: str,
    *,
    max_length: int = CHUNK_MAX_LENGTH,
    stride: int = CHUNK_STRIDE,
    max_chunks: int = MAX_SELECTED_CHUNKS,
) -> list[dict[str, object]]:
    """Tokenize with the tokenizer overflow mechanism and select representative chunks."""
    if max_length < 3:
        raise ValueError("max_length must leave room for special tokens")
    if stride < 0 or stride >= max_length:
        raise ValueError("stride must be between 0 and max_length - 1")
    if max_chunks < 1:
        raise ValueError("max_chunks must be positive")

    encoded = tokenizer(
        str(text or ""),
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        padding=False,
    )
    input_ids = encoded.get("input_ids", [])
    attention_masks = encoded.get("attention_mask", [])
    if input_ids and isinstance(input_ids[0], int):
        input_ids = [input_ids]
        attention_masks = [attention_masks]

    total_chunks = len(input_ids)
    selected_indices = select_representative_indices(total_chunks, maximum=max_chunks)
    chunks: list[dict[str, object]] = []
    for selected_position, source_index in enumerate(selected_indices):
        chunk: dict[str, object] = {
            "chunk_index": source_index,
            "selected_position": selected_position,
            "total_chunks": total_chunks,
            "input_ids": list(input_ids[source_index]),
            "attention_mask": list(attention_masks[source_index]),
        }
        for key in ("token_type_ids",):
            values = encoded.get(key)
            if values:
                chunk[key] = list(values[source_index])
        chunks.append(chunk)
    return chunks


def aggregate_scores(
    scores: torch.Tensor | Sequence[Sequence[float]],
    *,
    method: str = DEFAULT_AGGREGATION_METHOD,
    top_k: int = DEFAULT_AGGREGATION_TOP_K,
    scores_are_logits: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate page/chunk scores and return aggregate scores plus probabilities."""
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"Unsupported aggregation method: {method}")
    tensor = torch.as_tensor(scores, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError("scores must have shape [items, classes]")

    if method == "mean":
        aggregated = tensor.mean(dim=0)
    elif method == "max":
        aggregated = tensor.max(dim=0).values
    else:
        count = min(max(1, int(top_k)), tensor.shape[0])
        aggregated = torch.topk(tensor, k=count, dim=0).values.mean(dim=0)

    if scores_are_logits:
        probabilities = torch.softmax(aggregated, dim=0)
    else:
        probabilities = aggregated.clamp_min(0)
        total = probabilities.sum()
        if total <= 0:
            probabilities = torch.full_like(probabilities, 1.0 / probabilities.numel())
        else:
            probabilities = probabilities / total
    return aggregated, probabilities


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_count: int
) -> dict[str, object]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    per_class: dict[int, dict[str, float | int]] = {}
    for label_index in range(class_count):
        tp = sum(t == label_index and p == label_index for t, p in zip(y_true, y_pred))
        fp = sum(t != label_index and p == label_index for t, p in zip(y_true, y_pred))
        fn = sum(t == label_index and p != label_index for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[label_index] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(t == label_index for t in y_true),
        }
    total = len(y_true)
    return {
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / total if total else 0.0,
        "macro_precision": sum(precisions) / class_count if class_count else 0.0,
        "macro_recall": sum(recalls) / class_count if class_count else 0.0,
        "macro_f1": sum(f1_scores) / class_count if class_count else 0.0,
        "per_class": per_class,
    }


def aggregate_document_predictions(
    logits_by_document: Mapping[str, Sequence[Sequence[float]] | torch.Tensor],
    labels_by_document: Mapping[str, int],
    *,
    class_count: int,
    method: str,
    top_k: int = DEFAULT_AGGREGATION_TOP_K,
) -> dict[str, object]:
    document_ids = sorted(logits_by_document)
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: dict[str, list[float]] = {}
    for document_id in document_ids:
        if document_id not in labels_by_document:
            raise ValueError(f"Missing label for document: {document_id}")
        _, probs = aggregate_scores(
            logits_by_document[document_id], method=method, top_k=top_k, scores_are_logits=True
        )
        probabilities[document_id] = probs.tolist()
        y_true.append(int(labels_by_document[document_id]))
        y_pred.append(int(probs.argmax().item()))
    metrics = classification_metrics(y_true, y_pred, class_count)
    metrics.update(
        {
            "document_ids": document_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "probabilities": probabilities,
            "aggregation_method": method,
            "aggregation_top_k": top_k,
            "documents_evaluated": len(document_ids),
        }
    )
    return metrics


def choose_aggregation_method(
    logits_by_document: Mapping[str, Sequence[Sequence[float]] | torch.Tensor],
    labels_by_document: Mapping[str, int],
    *,
    class_count: int,
    methods: Sequence[str] = AGGREGATION_METHODS,
    top_k: int = DEFAULT_AGGREGATION_TOP_K,
) -> tuple[str, dict[str, dict[str, object]]]:
    results = {
        method: aggregate_document_predictions(
            logits_by_document,
            labels_by_document,
            class_count=class_count,
            method=method,
            top_k=top_k,
        )
        for method in methods
    }
    priority = {
        method: (1 if method == DEFAULT_AGGREGATION_METHOD else 0, -index)
        for index, method in enumerate(methods)
    }
    best = max(
        methods,
        key=lambda method: (
            float(results[method]["macro_f1"]),
            float(results[method]["accuracy"]),
            priority[method],
        ),
    )
    return best, results


def document_balanced_weights(
    rows: Sequence[Mapping[str, object]], document_key: str = "document_id"
) -> list[float]:
    counts = Counter(str(row.get(document_key, "")) for row in rows)
    if "" in counts:
        raise ValueError(f"Rows contain an empty {document_key}")
    return [1.0 / counts[str(row[document_key])] for row in rows]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _normalized_path(value: object) -> str:
    return str(value or "").replace("\\", "/").strip().casefold()


def validate_document_manifest(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = [dict(row) for row in rows]
    required = {
        "document_id",
        "parent_document_id",
        "augmentation_group_id",
        "label",
        "raw_path",
        "split",
    }
    if not rows:
        raise ValueError("Document manifest is empty")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Document manifest is missing columns: {', '.join(sorted(missing))}")

    ids = [str(row.get("document_id", "")).strip() for row in rows]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if not key or count > 1)
    if duplicate_ids:
        raise ManifestLeakageError(
            "Duplicate or empty document_id values: " + ", ".join(duplicate_ids[:10])
        )

    invalid_splits = sorted(
        {str(row.get("split", "")) for row in rows if row.get("split") not in VALID_SPLITS}
    )
    if invalid_splits:
        raise ValueError(f"Invalid split values: {', '.join(invalid_splits)}")

    violations: list[str] = []
    fields = (
        ("parent_document_id", lambda row: str(row.get("parent_document_id", "")).strip()),
        ("augmentation_group_id", lambda row: str(row.get("augmentation_group_id", "")).strip()),
        ("raw_path", lambda row: _normalized_path(row.get("raw_path", ""))),
        ("raw_sha256", lambda row: str(row.get("raw_sha256", "")).strip().casefold()),
    )
    for field, getter in fields:
        split_sets: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            key = getter(row)
            if key:
                split_sets[key].add(str(row["split"]))
        leaking = [(key, splits) for key, splits in split_sets.items() if len(splits) > 1]
        for key, splits in leaking[:20]:
            violations.append(f"{field}={key} occurs in {','.join(sorted(splits))}")

    if violations:
        details = "\n".join(f"  - {item}" for item in violations[:50])
        raise ManifestLeakageError(
            f"Cross-split leakage detected before training ({len(violations)} examples):\n{details}"
        )

    counts = Counter(str(row["split"]) for row in rows)
    label_counts: dict[str, dict[str, int]] = {}
    for split in VALID_SPLITS:
        label_counts[split] = dict(
            Counter(str(row["label"]) for row in rows if row["split"] == split)
        )
    return {
        "documents": len(rows),
        "split_counts": dict(counts),
        "label_counts": label_counts,
        "leakage": False,
    }


def validate_artifact_rows(
    document_rows: Sequence[Mapping[str, object]],
    artifact_rows: Sequence[Mapping[str, object]],
    *,
    artifact_name: str,
) -> dict[str, int]:
    document_by_id = {str(row["document_id"]): row for row in document_rows}
    counts = Counter()
    for artifact in artifact_rows:
        document_id = str(artifact.get("document_id", ""))
        document = document_by_id.get(document_id)
        if document is None:
            raise ValueError(f"{artifact_name} references unknown document: {document_id}")
        for field in ("parent_document_id", "augmentation_group_id", "label", "split"):
            if str(artifact.get(field, "")) != str(document.get(field, "")):
                raise ManifestLeakageError(
                    f"{artifact_name} {document_id} has mismatched {field}: "
                    f"{artifact.get(field)!r} != {document.get(field)!r}"
                )
        counts[document_id] += 1
    return {
        "documents_with_artifacts": len(counts),
        "artifacts": sum(counts.values()),
        "max_artifacts_per_document": max(counts.values(), default=0),
    }
