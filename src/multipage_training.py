from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import WeightedRandomSampler

try:
    from .multipage import (
        DEFAULT_AGGREGATION_METHOD,
        DEFAULT_AGGREGATION_TOP_K,
        aggregate_document_predictions,
        choose_aggregation_method,
        document_balanced_weights,
        read_csv_rows,
        validate_artifact_rows,
        validate_document_manifest,
    )
except ImportError:
    from multipage import (  # type: ignore
        DEFAULT_AGGREGATION_METHOD,
        DEFAULT_AGGREGATION_TOP_K,
        aggregate_document_predictions,
        choose_aggregation_method,
        document_balanced_weights,
        read_csv_rows,
        validate_artifact_rows,
        validate_document_manifest,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTIPAGE_DIR = PROJECT_ROOT / "data" / "multipage"
DOCUMENT_MANIFEST_PATH = MULTIPAGE_DIR / "document_manifest.csv"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def load_multipage_training_rows(
    artifact_kind: str, *, smoke_test: bool = False
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, object]]], Path]:
    if artifact_kind not in {"page", "resnet_page", "layout_page", "chunk"}:
        raise ValueError(f"Unsupported artifact kind: {artifact_kind}")
    if not DOCUMENT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing authoritative document manifest: {DOCUMENT_MANIFEST_PATH}. "
            "Run scripts/build_multipage_dataset.py first."
        )

    documents = read_csv_rows(DOCUMENT_MANIFEST_PATH)
    validate_document_manifest(documents)
    artifact_root = MULTIPAGE_DIR / "smoke_test" if smoke_test else MULTIPAGE_DIR
    is_page_artifact = artifact_kind in {"page", "resnet_page", "layout_page"}
    artifact_path = artifact_root / (
        "page_manifest.csv" if is_page_artifact else "chunk_manifest.jsonl"
    )
    if not artifact_path.is_file():
        mode_hint = " --smoke-test" if smoke_test else ""
        raise FileNotFoundError(
            f"Missing {artifact_kind} artifacts: {artifact_path}. Run: "
            f"python scripts/build_multipage_dataset.py{mode_hint} --skip-existing"
        )
    artifacts: list[dict[str, object]]
    if is_page_artifact:
        artifacts = [dict(row) for row in read_csv_rows(artifact_path)]
    else:
        artifacts = read_jsonl(artifact_path)

    if artifact_kind == "layout_page":
        artifacts = [
            row
            for row in artifacts
            if str(row.get("layout_status", "valid")) == "valid"
        ]
    elif artifact_kind == "resnet_page":
        artifacts = [row for row in artifacts if str(row.get("image_path", "")).strip()]

    selected_ids = {str(row.get("document_id", "")) for row in artifacts}
    selected_documents = [row for row in documents if row["document_id"] in selected_ids]
    if not selected_documents:
        raise ValueError(f"No document rows match artifacts in {artifact_path}")
    validate_artifact_rows(selected_documents, artifacts, artifact_name=artifact_kind)

    artifact_counts = Counter(str(row["document_id"]) for row in artifacts)
    missing = [row["document_id"] for row in selected_documents if artifact_counts[row["document_id"]] == 0]
    if missing:
        raise ValueError(f"Documents without {artifact_kind} artifacts: {', '.join(missing[:20])}")
    excluded_documents = len(documents) - len(selected_documents)
    if excluded_documents:
        print(
            f"WARNING: {excluded_documents} documents have no usable {artifact_kind} artifacts "
            "and will be excluded from this model only."
        )

    by_split: dict[str, list[dict[str, object]]] = {name: [] for name in ("train", "validation", "test")}
    for row in artifacts:
        by_split[str(row["split"])].append(row)
    for split, rows in by_split.items():
        if not rows:
            raise ValueError(f"No {artifact_kind} artifacts found for split: {split}")
    return selected_documents, by_split, artifact_path


def make_document_balanced_sampler(rows: Sequence[Mapping[str, object]]) -> WeightedRandomSampler:
    weights = torch.tensor(document_balanced_weights(rows), dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)


def append_batch_logits(
    logits_by_document: dict[str, list[list[float]]],
    labels_by_document: dict[str, int],
    document_ids: Sequence[str],
    labels: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    cpu_logits = logits.detach().float().cpu().tolist()
    cpu_labels = labels.detach().cpu().tolist()
    for document_id, label, item_logits in zip(document_ids, cpu_labels, cpu_logits):
        document_id = str(document_id)
        existing = labels_by_document.setdefault(document_id, int(label))
        if existing != int(label):
            raise ValueError(f"Document {document_id} has inconsistent labels")
        logits_by_document[document_id].append(item_logits)


def aggregate_evaluation(
    logits_by_document: Mapping[str, Sequence[Sequence[float]]],
    labels_by_document: Mapping[str, int],
    *,
    class_count: int,
    method: str | None = None,
    select_on_validation: bool = False,
    top_k: int = DEFAULT_AGGREGATION_TOP_K,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if select_on_validation:
        selected, comparisons = choose_aggregation_method(
            logits_by_document,
            labels_by_document,
            class_count=class_count,
            top_k=top_k,
        )
        return comparisons[selected], comparisons
    selected = method or DEFAULT_AGGREGATION_METHOD
    result = aggregate_document_predictions(
        logits_by_document,
        labels_by_document,
        class_count=class_count,
        method=selected,
        top_k=top_k,
    )
    return result, {selected: result}


def serializable_aggregation_comparison(
    comparisons: Mapping[str, Mapping[str, object]]
) -> dict[str, dict[str, object]]:
    keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "documents_evaluated")
    return {
        method: {key: metrics.get(key) for key in keys}
        for method, metrics in comparisons.items()
    }


def save_aggregation_config(
    path: Path,
    *,
    method: str,
    top_k: int,
    validation_comparisons: Mapping[str, Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "top_k": top_k,
        "selected_by": "validation_macro_f1",
        "validation_metrics": serializable_aggregation_comparison(validation_comparisons),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_aggregation_config(path: Path) -> tuple[str, int]:
    if not Path(path).is_file():
        return DEFAULT_AGGREGATION_METHOD, DEFAULT_AGGREGATION_TOP_K
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(payload.get("method", DEFAULT_AGGREGATION_METHOD)), int(
        payload.get("top_k", DEFAULT_AGGREGATION_TOP_K)
    )


def split_document_counts(rows_by_split: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, int]:
    return {
        split: len({str(row["document_id"]) for row in rows})
        for split, rows in rows_by_split.items()
    }
