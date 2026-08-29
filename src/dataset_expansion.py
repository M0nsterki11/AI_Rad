"""Shared safety utilities for real-world dataset expansion.

The expansion scripts use this module to fingerprint documents, reject duplicate
content, and create split assignments that keep augmentations with their source.
"""

from __future__ import annotations

import csv
import hashlib
import html
import math
import os
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import fitz
import numpy as np
from PIL import Image


CLASS_NAMES = ("invoice", "cv", "contract", "email", "scientific")
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".html", ".htm", ".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
METADATA_FIELDS = ("id", "label", "raw_path", "image_path", "text_path", "ocr_path")


class DatasetExpansionError(RuntimeError):
    """Raised when expansion safety validation fails."""


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def relative_project_path(project_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def validate_training_path(project_root: Path, path: Path) -> None:
    """Reject any path under either external evaluation directory."""
    resolved = path.resolve()
    forbidden = (
        (project_root / "data" / "external_test").resolve(),
        (project_root / "data" / "external_robus_test").resolve(),
    )
    for root in forbidden:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        raise DatasetExpansionError(f"External test path cannot be used for training: {path}")


def read_csv_rows(path: Path, required_fields: Sequence[str] | None = None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in (required_fields or ()) if field not in fieldnames]
        if missing:
            raise DatasetExpansionError(f"CSV {path} is missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def atomic_write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold().replace("\x00", " ")
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def visible_html_text(raw_html: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            with fitz.open(path) as document:
                return "\n".join(page.get_text("text") for page in document).strip()
        if suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        if suffix in {".html", ".htm"}:
            return visible_html_text(path.read_text(encoding="utf-8", errors="ignore"))
        if suffix == ".docx":
            from docx import Document

            document = Document(path)
            return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    except Exception:
        return ""
    return ""


def document_preview(path: Path) -> Image.Image | None:
    suffix = path.suffix.lower()
    try:
        if suffix in IMAGE_EXTENSIONS:
            with Image.open(path) as image:
                return image.convert("RGB").copy()
        if suffix == ".pdf":
            with fitz.open(path) as document:
                if document.page_count < 1:
                    return None
                pixmap = document.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    except Exception:
        return None
    return None


def perceptual_hash(image: Image.Image, hash_size: int = 8, high_frequency_factor: int = 4) -> int:
    """Return a pHash compatible 64-bit integer without an imagehash dependency."""
    size = hash_size * high_frequency_factor
    pixels = np.asarray(image.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float64)
    coordinates = np.arange(size, dtype=np.float64)
    frequencies = np.arange(hash_size, dtype=np.float64)[:, None]
    transform = np.cos((math.pi / size) * (coordinates + 0.5) * frequencies)
    coefficients = transform @ pixels @ transform.T
    low_frequency = coefficients[:hash_size, :hash_size]
    median = float(np.median(low_frequency.flatten()[1:]))
    bits = low_frequency > median
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def simhash(text: str, bits: int = 64) -> int | None:
    tokens = normalize_text(text).split()
    if not tokens:
        return None
    features = tokens if len(tokens) < 3 else [" ".join(tokens[index:index + 3]) for index in range(len(tokens) - 2)]
    weights = [0] * bits
    for feature in features:
        digest = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(bits):
            weights[bit] += 1 if digest & (1 << bit) else -1
    value = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            value |= 1 << bit
    return value


def bit_similarity(left: int, right: int, bits: int = 64) -> float:
    return 1.0 - ((left ^ right).bit_count() / bits)


def token_set(text: str, limit: int = 5000) -> frozenset[str]:
    tokens = normalize_text(text).split()
    if len(tokens) > limit:
        step = max(1, len(tokens) // limit)
        tokens = tokens[::step][:limit]
    return frozenset(tokens)


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(slots=True)
class FingerprintRecord:
    key: str
    path: str
    label: str
    source: str
    exact_sha256: str
    normalized_text_sha256: str | None = None
    text_simhash: int | None = None
    text_tokens: frozenset[str] = field(default_factory=frozenset)
    text_length: int = 0
    image_phash: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    group_id: str | None = None


@dataclass(slots=True)
class DuplicateMatch:
    reason: str
    similar_to: str
    similarity_score: float


def build_fingerprint_record(
    *,
    key: str,
    raw_path: Path,
    label: str,
    source: str,
    image_path: Path | None = None,
    text_path: Path | None = None,
    text_hint: str | None = None,
    group_id: str | None = None,
) -> FingerprintRecord:
    text = str(text_hint or "")
    if not text and text_path and text_path.exists():
        text = text_path.read_text(encoding="utf-8", errors="ignore")
    if not text:
        text = extract_document_text(raw_path)
    normalized = normalize_text(text)

    preview_source = image_path if image_path and image_path.exists() else raw_path
    preview = document_preview(preview_source)
    image_hash = None
    width = None
    height = None
    if preview is not None:
        width, height = preview.size
        image_hash = perceptual_hash(preview)
        preview.close()

    return FingerprintRecord(
        key=key,
        path=str(raw_path),
        label=label,
        source=source,
        exact_sha256=sha256_file(raw_path),
        normalized_text_sha256=(
            hashlib.sha256(normalized.encode("utf-8")).hexdigest() if len(normalized) >= 40 else None
        ),
        text_simhash=simhash(normalized) if len(normalized) >= 80 else None,
        text_tokens=token_set(normalized),
        text_length=len(normalized),
        image_phash=image_hash,
        image_width=width,
        image_height=height,
        group_id=group_id or key,
    )


class FingerprintIndex:
    def __init__(self, image_threshold: float = 0.94, text_threshold: float = 0.95):
        self.image_threshold = image_threshold
        self.text_threshold = text_threshold
        self.records: list[FingerprintRecord] = []
        self.exact: dict[str, list[FingerprintRecord]] = defaultdict(list)
        self.normalized_text: dict[str, list[FingerprintRecord]] = defaultdict(list)
        self.with_images: list[FingerprintRecord] = []
        self.with_text: list[FingerprintRecord] = []

    def add(self, record: FingerprintRecord) -> None:
        self.records.append(record)
        self.exact[record.exact_sha256].append(record)
        if record.normalized_text_sha256:
            self.normalized_text[record.normalized_text_sha256].append(record)
        if record.image_phash is not None:
            self.with_images.append(record)
        if record.text_simhash is not None:
            self.with_text.append(record)

    @staticmethod
    def _first_allowed(records: Iterable[FingerprintRecord], ignored: set[str]) -> FingerprintRecord | None:
        return next((record for record in records if record.key not in ignored and record.path not in ignored), None)

    def find_duplicate(self, candidate: FingerprintRecord, ignore: Iterable[str] = ()) -> DuplicateMatch | None:
        ignored = set(ignore)
        exact = self._first_allowed(self.exact.get(candidate.exact_sha256, ()), ignored)
        if exact:
            return DuplicateMatch("identical_sha256", exact.path, 1.0)

        if candidate.normalized_text_sha256:
            exact_text = self._first_allowed(
                self.normalized_text.get(candidate.normalized_text_sha256, ()), ignored
            )
            if exact_text:
                return DuplicateMatch("identical_normalized_text", exact_text.path, 1.0)

        if candidate.image_phash is not None:
            for existing in self.with_images:
                if existing.key in ignored or existing.path in ignored or existing.image_phash is None:
                    continue
                visual_score = bit_similarity(candidate.image_phash, existing.image_phash)
                if visual_score < self.image_threshold:
                    continue
                content_score = jaccard_similarity(candidate.text_tokens, existing.text_tokens)
                if candidate.text_tokens and existing.text_tokens:
                    # Document templates often have almost identical pHashes even
                    # when their actual text differs. Treat the visual match as a
                    # duplicate only when textual content confirms it.
                    if content_score < 0.82:
                        continue
                    score = min(visual_score, content_score)
                else:
                    # Image-only files have no textual evidence, so require an
                    # extremely close visual match before rejecting them.
                    if visual_score < 0.985:
                        continue
                    score = visual_score
                return DuplicateMatch("near_identical_first_page", existing.path, score)

        if candidate.text_simhash is not None:
            for existing in self.with_text:
                if existing.key in ignored or existing.path in ignored or existing.text_simhash is None:
                    continue
                simhash_score = bit_similarity(candidate.text_simhash, existing.text_simhash)
                if simhash_score < self.text_threshold:
                    continue
                token_score = jaccard_similarity(candidate.text_tokens, existing.text_tokens)
                if token_score < 0.80:
                    continue
                return DuplicateMatch("near_duplicate_text", existing.path, min(simhash_score, token_score))
        return None


def load_existing_fingerprint_index(
    project_root: Path,
    metadata_rows: Sequence[Mapping[str, str]],
    *,
    image_threshold: float = 0.94,
    text_threshold: float = 0.95,
) -> tuple[FingerprintIndex, list[dict[str, str]]]:
    index = FingerprintIndex(image_threshold=image_threshold, text_threshold=text_threshold)
    failures: list[dict[str, str]] = []
    tracked_raw_paths: set[Path] = set()
    for row in metadata_rows:
        raw_path = project_path(project_root, row.get("raw_path", ""))
        image_path = project_path(project_root, row.get("image_path", ""))
        text_path = project_path(project_root, row.get("text_path", ""))
        try:
            validate_training_path(project_root, raw_path)
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)
            tracked_raw_paths.add(raw_path.resolve())
            record = build_fingerprint_record(
                key=row.get("id", str(raw_path)),
                raw_path=raw_path,
                image_path=image_path,
                text_path=text_path,
                label=row.get("label", ""),
                source="existing_dataset",
            )
            index.add(record)
        except Exception as error:
            failures.append(
                {
                    "id": row.get("id", ""),
                    "raw_path": str(raw_path),
                    "error": str(error),
                }
            )

    raw_root = project_root / "data" / "raw"
    for label in CLASS_NAMES:
        label_root = raw_root / label
        if not label_root.exists():
            continue
        for raw_path in sorted(label_root.rglob("*")):
            if (
                not raw_path.is_file()
                or raw_path.suffix.lower() not in SUPPORTED_EXTENSIONS
                or raw_path.resolve() in tracked_raw_paths
            ):
                continue
            try:
                validate_training_path(project_root, raw_path)
                record = build_fingerprint_record(
                    key=f"untracked:{raw_path}",
                    raw_path=raw_path,
                    label=label,
                    source="untracked_data_raw",
                )
                index.add(record)
            except Exception as error:
                failures.append(
                    {
                        "id": "",
                        "raw_path": str(raw_path),
                        "error": f"untracked raw file: {error}",
                    }
                )

    # Holdout files are exclusion fingerprints only; they are never returned as
    # metadata rows and therefore can never enter a training split.
    for holdout_name in ("external_test", "external_robus_test"):
        holdout_root = project_root / "data" / holdout_name
        if not holdout_root.exists():
            continue
        for label in CLASS_NAMES:
            label_root = holdout_root / label
            if not label_root.exists():
                continue
            for holdout_path in sorted(label_root.rglob("*")):
                if not holdout_path.is_file() or holdout_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                try:
                    record = build_fingerprint_record(
                        key=f"holdout:{holdout_path}",
                        raw_path=holdout_path,
                        label=label,
                        source=f"{holdout_name}_guard",
                    )
                    index.add(record)
                except Exception as error:
                    failures.append(
                        {
                            "id": "",
                            "raw_path": str(holdout_path),
                            "error": f"holdout guard: {error}",
                        }
                    )
    return index, failures


def _resolve_parent(document_id: str, augmentation_parents: Mapping[str, str]) -> str:
    seen: set[str] = set()
    current = document_id
    while current in augmentation_parents and current not in seen:
        seen.add(current)
        current = augmentation_parents[current]
    return current


def group_aware_stratified_split(
    rows: Sequence[Mapping[str, str]],
    augmentation_parents: Mapping[str, str],
    *,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Create 75/15/10 label-stratified splits while preserving related groups."""
    output = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)

    for label in CLASS_NAMES:
        label_rows = [dict(row) for row in rows if row.get("label") == label]
        if len(label_rows) < 3:
            raise DatasetExpansionError(f"Class {label} has fewer than three documents.")

        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in label_rows:
            document_id = row.get("id", "")
            groups[_resolve_parent(document_id, augmentation_parents)].append(row)

        shuffled = list(groups.items())
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda item: len(item[1]), reverse=True)

        total = len(label_rows)
        targets = {
            "train": int(total * 0.75),
            "validation": int(total * 0.15),
        }
        targets["test"] = total - targets["train"] - targets["validation"]
        counts = {name: 0 for name in output}

        for _, group_rows in shuffled:
            group_size = len(group_rows)
            ranked = sorted(
                output,
                key=lambda name: (
                    (targets[name] - counts[name]) / max(targets[name], 1),
                    -counts[name],
                    rng.random(),
                ),
                reverse=True,
            )
            fitting = [name for name in ranked if counts[name] + group_size <= targets[name]]
            destination = fitting[0] if fitting else ranked[0]
            output[destination].extend(group_rows)
            counts[destination] += group_size

    for rows_for_split in output.values():
        rng.shuffle(rows_for_split)

    split_by_id = {
        row["id"]: split_name
        for split_name, split_rows in output.items()
        for row in split_rows
    }
    for child, parent in augmentation_parents.items():
        if child in split_by_id and parent in split_by_id and split_by_id[child] != split_by_id[parent]:
            raise DatasetExpansionError(f"Augmentation split leak: {child} and {parent}")
    return output


def class_counts(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    counts = {label: 0 for label in CLASS_NAMES}
    for row in rows:
        label = row.get("label", "")
        if label in counts:
            counts[label] += 1
    return counts
