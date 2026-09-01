from __future__ import annotations

import csv
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .dataset_expansion import METADATA_FIELDS, group_aware_stratified_split
from .multipage import validate_document_manifest


DOCUMENT_MANIFEST_FIELDS = (
    "document_id",
    "parent_document_id",
    "augmentation_group_id",
    "label",
    "raw_path",
    "image_path",
    "text_path",
    "ocr_path",
    "raw_sha256",
    "split",
)


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        canonical, other = sorted((left_root, right_root))
        self.parent[other] = canonical


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def atomic_write_csv(
    path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def augmentation_parent_map(
    source_rows: Iterable[Mapping[str, str]], document_ids: set[str]
) -> dict[str, str]:
    parents: dict[str, str] = {}
    for row in source_rows:
        child = str(row.get("id", "")).strip()
        parent = str(row.get("original_id", "")).strip()
        is_augmented = str(row.get("is_augmented", "")).strip().casefold()
        if is_augmented in {"true", "1", "yes"} and child in document_ids and parent in document_ids:
            parents[child] = parent
    return parents


def build_document_manifest(
    project_root: Path,
    metadata_rows: Sequence[Mapping[str, str]],
    source_rows: Sequence[Mapping[str, str]],
    *,
    seed: int = 42,
    compute_hashes: bool = True,
) -> list[dict[str, str]]:
    project_root = Path(project_root).resolve()
    metadata = [{field: str(row.get(field, "") or "") for field in METADATA_FIELDS} for row in metadata_rows]
    document_ids = {row["id"] for row in metadata}
    if len(document_ids) != len(metadata):
        raise ValueError("metadata contains duplicate document IDs")

    immediate_parents = augmentation_parent_map(source_rows, document_ids)
    union_find = UnionFind(document_ids)
    for child, parent in immediate_parents.items():
        union_find.union(child, parent)

    hashes: dict[str, str] = {}
    ids_by_hash: dict[str, list[str]] = defaultdict(list)
    for row in metadata:
        raw_path = Path(row["raw_path"])
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw document is missing: {raw_path}")
        digest = sha256_file(raw_path) if compute_hashes else ""
        hashes[row["id"]] = digest
        if digest:
            ids_by_hash[digest].append(row["id"])

    labels_by_id = {row["id"]: row["label"] for row in metadata}
    for digest, matching_ids in ids_by_hash.items():
        matching_labels = {labels_by_id[document_id] for document_id in matching_ids}
        if len(matching_labels) > 1:
            raise ValueError(
                f"Identical raw SHA-256 occurs under different labels: {digest} -> "
                f"{', '.join(sorted(matching_labels))}"
            )
        anchor = matching_ids[0]
        for document_id in matching_ids[1:]:
            union_find.union(anchor, document_id)

    group_parent_map = {
        document_id: union_find.find(document_id)
        for document_id in document_ids
        if union_find.find(document_id) != document_id
    }
    split_rows = group_aware_stratified_split(metadata, group_parent_map, seed=seed)
    split_by_id = {
        row["id"]: split_name
        for split_name, rows in split_rows.items()
        for row in rows
    }

    manifest: list[dict[str, str]] = []
    for row in metadata:
        document_id = row["id"]
        manifest.append(
            {
                "document_id": document_id,
                "parent_document_id": immediate_parents.get(document_id, document_id),
                "augmentation_group_id": union_find.find(document_id),
                "label": row["label"],
                "raw_path": row["raw_path"],
                "image_path": row["image_path"],
                "text_path": row["text_path"],
                "ocr_path": row["ocr_path"],
                "raw_sha256": hashes[document_id],
                "split": split_by_id[document_id],
            }
        )

    validate_document_manifest(manifest)
    return manifest


def write_document_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    validate_document_manifest(rows)
    atomic_write_csv(path, rows, DOCUMENT_MANIFEST_FIELDS)
