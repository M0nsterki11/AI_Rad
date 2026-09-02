from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_expansion import (  # noqa: E402
    METADATA_FIELDS,
    FingerprintIndex,
    atomic_write_csv,
    build_fingerprint_record,
    extract_document_text,
    load_existing_fingerprint_index,
    project_path,
    read_csv_rows,
    relative_project_path,
    sha256_file,
)
from src.multipage import (  # noqa: E402
    CHUNK_MAX_LENGTH,
    CHUNK_STRIDE,
    MAX_SELECTED_CHUNKS,
    tokenize_document_chunks,
    validate_artifact_rows,
    validate_document_manifest,
)
from src.multipage_manifest import (  # noqa: E402
    DOCUMENT_MANIFEST_FIELDS,
)
from src.multipage_preprocess import (  # noqa: E402
    LAYOUT_STATUS_VALID,
    PAGE_MANIFEST_FIELDS,
    prepare_document_page_artifacts,
)
from src.preprocess import (  # noqa: E402
    MIN_TEXT_CHARS,
    clean_text,
    process_file_to_outputs,
    run_ocr_on_image,
)


DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CANDIDATE_ROOT = DATA_DIR / "Second_Traning_data"
METADATA_PATH = DATA_DIR / "metadata.csv"
SOURCE_TRACKING_PATH = DATA_DIR / "dataset_sources_extra.csv"
SPLITS_DIR = DATA_DIR / "splits"
MULTIPAGE_DIR = DATA_DIR / "multipage"
DOCUMENT_MANIFEST_PATH = MULTIPAGE_DIR / "document_manifest.csv"
PAGE_MANIFEST_PATH = MULTIPAGE_DIR / "page_manifest.csv"
CHUNK_MANIFEST_PATH = MULTIPAGE_DIR / "chunk_manifest.jsonl"
STAGING_ROOT = DATA_DIR / "staging_second_training_import"
RESULTS_DIR = PROJECT_ROOT / "results" / "dataset_expansion"
TOKENIZER_CANDIDATES = (
    PROJECT_ROOT / "models" / "xlm_roberta_multipage" / "best_model",
    PROJECT_ROOT / "models" / "xlm_roberta" / "best_model",
)

CLASS_ALIASES = {
    "contracts": "contract",
    "contract": "contract",
    "cv": "cv",
    "email": "email",
    "invoice": "invoice",
    "znanstveni rad": "scientific",
    "scientific": "scientific",
}
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
SOURCE_FIELDS = (
    "id",
    "label",
    "raw_path",
    "source_name",
    "source_url_or_dataset",
    "download_date",
    "original_id",
    "language",
    "is_synthetic",
    "is_augmented",
    "augmentation_type",
    "duplicate_check_status",
)
DUPLICATE_REPORT_FIELDS = (
    "candidate_path",
    "label",
    "document_id",
    "decision",
    "reason",
    "similar_to",
    "similarity_score",
    "sha256",
)
IMPORT_MANIFEST_FIELDS = (
    "id",
    "label",
    "source_path",
    "raw_path",
    "image_path",
    "text_path",
    "ocr_path",
    "sha256",
    "split",
)
FAILURE_FIELDS = (
    "candidate_path",
    "label",
    "document_id",
    "stage",
    "reason",
    "error_message",
    "elapsed_seconds",
)
FAILED_PAGE_FIELDS = (
    "document_id",
    "label",
    "raw_path",
    "page_index",
    "page_number",
    "status",
    "reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import local second-training documents safely into the existing "
            "multi-page dataset. New documents are assigned only to train."
        )
    )
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--commit", action="store_true", help="Apply the audited import.")
    parser.add_argument("--per-document-timeout", type=int, default=120)
    parser.add_argument("--image-similarity-threshold", type=float, default=0.94)
    parser.add_argument("--text-similarity-threshold", type=float, default=0.95)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.per_document_timeout < 5:
        parser.error("--per-document-timeout must be at least 5 seconds")
    return args


def worker_main(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    process_file_to_outputs(
        Path(job["raw_path"]),
        str(job["label"]),
        Path(job["image_path"]),
        Path(job["text_path"]),
        Path(job["ocr_path"]),
    )


def resolve_candidate_root(path: Path) -> Path:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Candidate directory does not exist: {path}")
    return path


def candidate_label(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) < 2:
        raise ValueError(f"Candidate is not inside a class folder: {relative}")
    folder = relative.parts[0].strip().casefold()
    if folder not in CLASS_ALIASES:
        raise ValueError(f"Unknown class folder '{relative.parts[0]}': {relative}")
    return CLASS_ALIASES[folder]


def collect_candidates(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    failures = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            failures.append(
                failure_row(relative, "", "", "inventory", "unsupported_extension", path.suffix)
            )
            continue
        try:
            label = candidate_label(path, root)
            digest = sha256_file(path)
        except Exception as error:
            failures.append(
                failure_row(relative, "", "", "inventory", "candidate_validation_failed", str(error))
            )
            continue
        candidates.append(
            {
                "path": path,
                "relative_path": relative,
                "label": label,
                "sha256": digest,
                "document_id": f"second_train_{label}_{digest[:12]}",
            }
        )
    return candidates, failures


def candidate_text_hint(path: Path) -> str:
    text = clean_text(extract_document_text(path))
    if len(text) >= 80:
        return text
    try:
        if path.suffix.casefold() in {".png", ".jpg", ".jpeg"}:
            with Image.open(path) as image:
                ocr_text, _ = run_ocr_on_image(image.convert("RGB"), "unknown", page_index=0)
            return clean_text(ocr_text)
        if path.suffix.casefold() == ".pdf":
            import fitz

            with fitz.open(path) as document:
                if document.page_count < 1:
                    return text
                page = document.load_page(0)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                try:
                    ocr_text, _ = run_ocr_on_image(image, "unknown", page_index=0)
                finally:
                    image.close()
                return clean_text(ocr_text)
    except Exception:
        return text
    return text


def audit_candidates(
    candidates: Sequence[dict[str, Any]],
    existing_metadata: Sequence[Mapping[str, str]],
    *,
    image_threshold: float,
    text_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    print(f"Building duplicate guard from {len(existing_metadata)} existing documents...")
    index, fingerprint_failures = load_existing_fingerprint_index(
        PROJECT_ROOT,
        existing_metadata,
        image_threshold=image_threshold,
        text_threshold=text_threshold,
    )
    decisions = []
    accepted = []
    failures = [
        failure_row(
            item.get("raw_path", ""),
            "",
            item.get("id", ""),
            "duplicate_index",
            "existing_fingerprint_failed",
            item.get("error", ""),
        )
        for item in fingerprint_failures
    ]
    existing_ids = {str(row["id"]) for row in existing_metadata}

    for candidate in tqdm(candidates, desc="Exact/near duplicate audit"):
        path = Path(candidate["path"])
        document_id = str(candidate["document_id"])
        if document_id in existing_ids:
            decisions.append(
                duplicate_row(candidate, "already_imported", "document_id_exists", document_id, 1.0)
            )
            continue
        try:
            text_hint = candidate_text_hint(path)
            fingerprint = build_fingerprint_record(
                key=f"second_training:{document_id}",
                raw_path=path,
                label=str(candidate["label"]),
                source="second_training_candidate",
                text_hint=text_hint,
                group_id=document_id,
            )
            # A prior interrupted commit may already have copied this exact file
            # to its deterministic destination without updating the manifests.
            # Ignore only that identical destination so a rerun can resume; all
            # other training and holdout matches remain duplicate guards.
            destination = final_paths(candidate)["raw"]
            ignored: list[str] = []
            if destination.is_file() and sha256_file(destination) == candidate["sha256"]:
                ignored.extend((str(destination), f"untracked:{destination}"))
            match = index.find_duplicate(fingerprint, ignore=ignored)
            if match:
                decisions.append(
                    duplicate_row(
                        candidate,
                        "duplicate",
                        match.reason,
                        match.similar_to,
                        match.similarity_score,
                    )
                )
            else:
                decisions.append(duplicate_row(candidate, "keep", "unique", "", ""))
                accepted.append({**candidate, "text_hint": text_hint})
            index.add(fingerprint)
        except Exception as error:
            decisions.append(
                duplicate_row(candidate, "error", "fingerprint_failed", "", "")
            )
            failures.append(
                failure_row(
                    candidate["relative_path"],
                    candidate["label"],
                    document_id,
                    "duplicate_audit",
                    "fingerprint_failed",
                    str(error),
                )
            )
    return accepted, decisions, failures


def duplicate_row(
    candidate: Mapping[str, Any],
    decision: str,
    reason: str,
    similar_to: str,
    score: float | str,
) -> dict[str, str]:
    return {
        "candidate_path": str(candidate["relative_path"]),
        "label": str(candidate["label"]),
        "document_id": str(candidate["document_id"]),
        "decision": decision,
        "reason": reason,
        "similar_to": str(similar_to),
        "similarity_score": f"{score:.6f}" if isinstance(score, float) else str(score),
        "sha256": str(candidate["sha256"]),
    }


def failure_row(
    path: str,
    label: str,
    document_id: str,
    stage: str,
    reason: str,
    error: str,
    elapsed: float | str = "",
) -> dict[str, str]:
    return {
        "candidate_path": str(path),
        "label": str(label),
        "document_id": str(document_id),
        "stage": str(stage),
        "reason": str(reason),
        "error_message": str(error),
        "elapsed_seconds": f"{elapsed:.3f}" if isinstance(elapsed, float) else str(elapsed),
    }


def final_paths(candidate: Mapping[str, Any]) -> dict[str, Path]:
    document_id = str(candidate["document_id"])
    label = str(candidate["label"])
    extension = Path(candidate["path"]).suffix.casefold()
    return {
        "raw": DATA_DIR / "raw" / label / "second_training" / f"{document_id}{extension}",
        "image": DATA_DIR / "processed" / "images" / f"{document_id}.png",
        "text": DATA_DIR / "processed" / "texts" / f"{document_id}.txt",
        "ocr": DATA_DIR / "processed" / "ocr" / f"{document_id}.json",
    }


def stage_paths(candidate: Mapping[str, Any]) -> dict[str, Path]:
    document_id = str(candidate["document_id"])
    root = STAGING_ROOT / document_id
    return {
        "root": root,
        "job": root / "job.json",
        "image": root / "image.png",
        "text": root / "text.txt",
        "ocr": root / "ocr.json",
    }


def validate_staged_outputs(paths: Mapping[str, Path]) -> None:
    for key in ("image", "text", "ocr"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"Missing staged {key}: {paths[key]}")
    with Image.open(paths["image"]) as image:
        image.verify()
    text = clean_text(paths["text"].read_text(encoding="utf-8", errors="ignore"))
    if len(text) < MIN_TEXT_CHARS:
        raise ValueError(f"Extracted text is too short: {len(text)} characters")
    payload = json.loads(paths["ocr"].read_text(encoding="utf-8"))
    if not isinstance(payload.get("words", []), list) or not isinstance(payload.get("boxes", []), list):
        raise ValueError("OCR JSON does not contain words and boxes lists")


def stage_candidate(candidate: dict[str, Any], timeout: int) -> tuple[dict[str, Any] | None, dict | None]:
    paths = stage_paths(candidate)
    paths["root"].mkdir(parents=True, exist_ok=True)
    job = {
        "raw_path": str(candidate["path"]),
        "label": candidate["label"],
        "image_path": str(paths["image"]),
        "text_path": str(paths["text"]),
        "ocr_path": str(paths["ocr"]),
    }
    paths["job"].write_text(json.dumps(job, indent=2), encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-job", str(paths["job"])]
    started = datetime.now()
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=creationflags,
            check=False,
        )
        elapsed = (datetime.now() - started).total_seconds()
    except subprocess.TimeoutExpired:
        elapsed = (datetime.now() - started).total_seconds()
        return None, failure_row(
            candidate["relative_path"],
            candidate["label"],
            candidate["document_id"],
            "preprocessing",
            "timeout",
            f"Document exceeded {timeout} seconds",
            elapsed,
        )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "worker failed").strip()
        return None, failure_row(
            candidate["relative_path"],
            candidate["label"],
            candidate["document_id"],
            "preprocessing",
            "worker_failed",
            message[-2000:],
            elapsed,
        )
    try:
        validate_staged_outputs(paths)
    except Exception as error:
        return None, failure_row(
            candidate["relative_path"],
            candidate["label"],
            candidate["document_id"],
            "preprocessing",
            "invalid_outputs",
            str(error),
            elapsed,
        )
    return {**candidate, "stage_paths": paths, "elapsed_seconds": elapsed}, None


def preprocess_candidates(
    candidates: Sequence[dict[str, Any]], timeout: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    prepared = []
    failures = []
    for candidate in tqdm(candidates, desc="Preprocessing second-training documents"):
        item, failure = stage_candidate(candidate, timeout)
        if failure:
            failures.append(failure)
            tqdm.write(f"SKIPPED {failure['reason']}: {candidate['relative_path']}")
        else:
            prepared.append(item)
    return prepared, failures


def backup_tables() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = PROJECT_ROOT / f"backup_second_training_{timestamp}"
    paths = (
        METADATA_PATH,
        SOURCE_TRACKING_PATH,
        SPLITS_DIR / "train.csv",
        SPLITS_DIR / "validation.csv",
        SPLITS_DIR / "test.csv",
        DOCUMENT_MANIFEST_PATH,
        PAGE_MANIFEST_PATH,
        CHUNK_MANIFEST_PATH,
    )
    for source in paths:
        if not source.is_file():
            continue
        relative = source.relative_to(PROJECT_ROOT)
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return backup_root


def copy_without_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(f"Destination exists with different content: {destination}")
        return
    shutil.copy2(source, destination)


def metadata_row(candidate: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        "id": str(candidate["document_id"]),
        "label": str(candidate["label"]),
        "raw_path": relative_project_path(PROJECT_ROOT, paths["raw"]),
        "image_path": relative_project_path(PROJECT_ROOT, paths["image"]),
        "text_path": relative_project_path(PROJECT_ROOT, paths["text"]),
        "ocr_path": relative_project_path(PROJECT_ROOT, paths["ocr"]),
    }


def source_row(candidate: Mapping[str, Any], row: Mapping[str, str]) -> dict[str, str]:
    return {
        "id": str(candidate["document_id"]),
        "label": str(candidate["label"]),
        "raw_path": row["raw_path"],
        "source_name": "manual_second_training",
        "source_url_or_dataset": f"local://data/Second_Traning_data/{candidate['relative_path']}",
        "download_date": date.today().isoformat(),
        "original_id": str(candidate["document_id"]),
        "language": "",
        "is_synthetic": "False",
        "is_augmented": "False",
        "augmentation_type": "",
        "duplicate_check_status": "passed",
    }


def document_manifest_row(row: Mapping[str, str], raw_sha256: str) -> dict[str, str]:
    document_id = str(row["id"])
    return {
        "document_id": document_id,
        "parent_document_id": document_id,
        "augmentation_group_id": document_id,
        "label": str(row["label"]),
        "raw_path": str(row["raw_path"]),
        "image_path": str(row["image_path"]),
        "text_path": str(row["text_path"]),
        "ocr_path": str(row["ocr_path"]),
        "raw_sha256": raw_sha256,
        "split": "train",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def tokenizer_path() -> Path:
    path = next((item for item in TOKENIZER_CANDIDATES if item.is_dir()), None)
    if path is None:
        raise FileNotFoundError("No local XLM-RoBERTa tokenizer directory was found")
    return path


def build_new_multipage_artifacts(
    document_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    page_rows = []
    chunk_rows = []
    failed_pages = []
    failures = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path(), local_files_only=True)

    for row in tqdm(document_rows, desc="Building new multi-page artifacts"):
        image_pages = 0
        valid_layout_pages = 0
        try:
            total_pages, selected_indices, artifacts = prepare_document_page_artifacts(
                PROJECT_ROOT,
                MULTIPAGE_DIR / "pages",
                row,
                skip_existing=True,
            )
            for artifact in artifacts:
                if artifact.image_path.is_file():
                    manifest = artifact.manifest_row(PROJECT_ROOT)
                    manifest["total_pages"] = total_pages
                    manifest["selected_page_count"] = len(selected_indices)
                    page_rows.append(manifest)
                    image_pages += 1
                if artifact.is_layout_valid and artifact.ocr_path.is_file():
                    valid_layout_pages += 1
                else:
                    failed_pages.append(
                        {
                            "document_id": row["document_id"],
                            "label": row["label"],
                            "raw_path": row["raw_path"],
                            "page_index": artifact.page_index,
                            "page_number": artifact.page_index + 1,
                            "status": artifact.layout_status,
                            "reason": artifact.failure_reason,
                        }
                    )
        except Exception as error:
            failures.append(
                failure_row(
                    row["raw_path"],
                    row["label"],
                    row["document_id"],
                    "multipage_pages",
                    "page_artifact_failed",
                    str(error),
                )
            )
        if image_pages == 0:
            failures.append(
                failure_row(
                    row["raw_path"],
                    row["label"],
                    row["document_id"],
                    "multipage_pages",
                    "no_resnet_pages",
                    "No renderable page image was created",
                )
            )
        if valid_layout_pages == 0:
            failures.append(
                failure_row(
                    row["raw_path"],
                    row["label"],
                    row["document_id"],
                    "multipage_pages",
                    "no_layout_pages",
                    "No page has aligned OCR words and boxes",
                )
            )

        try:
            text_path = project_path(PROJECT_ROOT, row["text_path"])
            text = text_path.read_text(encoding="utf-8", errors="ignore")
            chunks = tokenize_document_chunks(
                tokenizer,
                text,
                max_length=CHUNK_MAX_LENGTH,
                stride=CHUNK_STRIDE,
                max_chunks=MAX_SELECTED_CHUNKS,
            )
            if not chunks:
                raise ValueError("Tokenizer produced no chunks")
            for chunk in chunks:
                chunk_rows.append(
                    {
                        "document_id": row["document_id"],
                        "parent_document_id": row["parent_document_id"],
                        "augmentation_group_id": row["augmentation_group_id"],
                        "label": row["label"],
                        "split": row["split"],
                        "chunk_index": chunk["chunk_index"],
                        "selected_position": chunk["selected_position"],
                        "total_chunks": chunk["total_chunks"],
                        "selected_chunk_count": len(chunks),
                        "input_ids": chunk["input_ids"],
                        "attention_mask": chunk["attention_mask"],
                    }
                )
        except Exception as error:
            failures.append(
                failure_row(
                    row["raw_path"],
                    row["label"],
                    row["document_id"],
                    "multipage_chunks",
                    "chunk_generation_failed",
                    str(error),
                )
            )
    return page_rows, chunk_rows, failed_pages, failures


def verify_existing_split_partition(
    metadata: Sequence[Mapping[str, str]],
    train: Sequence[Mapping[str, str]],
    validation: Sequence[Mapping[str, str]],
    test: Sequence[Mapping[str, str]],
) -> None:
    metadata_ids = {str(row["id"]) for row in metadata}
    split_sets = [
        {str(row["id"]) for row in rows}
        for rows in (train, validation, test)
    ]
    if set().union(*split_sets) != metadata_ids:
        raise ValueError("Existing legacy splits do not contain exactly all metadata IDs")
    if any(split_sets[left] & split_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Existing legacy train/validation/test ID sets overlap")


def commit_import(
    prepared: Sequence[dict[str, Any]],
    existing_metadata: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], Path, list[dict[str, str]], list[dict[str, str]]]:
    existing_sources = read_csv_rows(SOURCE_TRACKING_PATH) if SOURCE_TRACKING_PATH.is_file() else []
    existing_train = read_csv_rows(SPLITS_DIR / "train.csv", METADATA_FIELDS)
    existing_validation = read_csv_rows(SPLITS_DIR / "validation.csv", METADATA_FIELDS)
    existing_test = read_csv_rows(SPLITS_DIR / "test.csv", METADATA_FIELDS)
    existing_documents = read_csv_rows(DOCUMENT_MANIFEST_PATH, DOCUMENT_MANIFEST_FIELDS)
    existing_pages = read_csv_rows(PAGE_MANIFEST_PATH, PAGE_MANIFEST_FIELDS)
    existing_chunks = read_jsonl(CHUNK_MANIFEST_PATH)
    verify_existing_split_partition(
        existing_metadata, existing_train, existing_validation, existing_test
    )
    validate_document_manifest(existing_documents)
    old_split_by_id = {row["document_id"]: row["split"] for row in existing_documents}

    backup_root = backup_tables()
    metadata_additions = []
    source_additions = []
    import_rows = []
    new_documents = []
    for candidate in prepared:
        final = final_paths(candidate)
        staged = candidate["stage_paths"]
        copy_without_overwrite(Path(candidate["path"]), final["raw"])
        copy_without_overwrite(staged["image"], final["image"])
        copy_without_overwrite(staged["text"], final["text"])
        copy_without_overwrite(staged["ocr"], final["ocr"])
        row = metadata_row(candidate, final)
        metadata_additions.append(row)
        source_additions.append(source_row(candidate, row))
        document = document_manifest_row(row, str(candidate["sha256"]))
        new_documents.append(document)
        import_rows.append(
            {
                "id": row["id"],
                "label": row["label"],
                "source_path": candidate["relative_path"],
                "raw_path": row["raw_path"],
                "image_path": row["image_path"],
                "text_path": row["text_path"],
                "ocr_path": row["ocr_path"],
                "sha256": candidate["sha256"],
                "split": "train",
            }
        )

    new_pages, new_chunks, failed_pages, artifact_failures = build_new_multipage_artifacts(
        new_documents
    )
    merged_metadata = existing_metadata + metadata_additions
    merged_train = existing_train + metadata_additions
    merged_documents = existing_documents + new_documents
    merged_pages = existing_pages + new_pages
    merged_chunks = existing_chunks + new_chunks

    verify_existing_split_partition(
        merged_metadata, merged_train, existing_validation, existing_test
    )
    validate_document_manifest(merged_documents)
    validate_artifact_rows(merged_documents, merged_pages, artifact_name="page manifest")
    validate_artifact_rows(merged_documents, merged_chunks, artifact_name="chunk manifest")
    if any(old_split_by_id[row["document_id"]] != row["split"] for row in existing_documents):
        raise RuntimeError("An existing document split changed unexpectedly")
    if any(row["split"] != "train" for row in new_documents):
        raise RuntimeError("A second-training document was assigned outside train")

    atomic_write_csv(METADATA_PATH, merged_metadata, METADATA_FIELDS)
    atomic_write_csv(SOURCE_TRACKING_PATH, existing_sources + source_additions, SOURCE_FIELDS)
    atomic_write_csv(SPLITS_DIR / "train.csv", merged_train, METADATA_FIELDS)
    atomic_write_csv(DOCUMENT_MANIFEST_PATH, merged_documents, DOCUMENT_MANIFEST_FIELDS)
    atomic_write_csv(PAGE_MANIFEST_PATH, merged_pages, PAGE_MANIFEST_FIELDS)
    atomic_write_jsonl(CHUNK_MANIFEST_PATH, merged_chunks)
    return import_rows, artifact_failures, backup_root, failed_pages, new_documents


def write_reports(
    decisions: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    import_rows: Sequence[Mapping[str, Any]],
    failed_pages: Sequence[Mapping[str, Any]],
    *,
    commit: bool,
    backup_root: Path | None,
    candidate_count: int,
    accepted_count: int,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(
        RESULTS_DIR / "second_training_duplicate_report.csv",
        decisions,
        DUPLICATE_REPORT_FIELDS,
    )
    atomic_write_csv(
        RESULTS_DIR / "second_training_import_failures.csv",
        failures,
        FAILURE_FIELDS,
    )
    atomic_write_csv(
        RESULTS_DIR / "second_training_import_manifest.csv",
        import_rows,
        IMPORT_MANIFEST_FIELDS,
    )
    atomic_write_csv(
        RESULTS_DIR / "second_training_failed_pages.csv",
        failed_pages,
        FAILED_PAGE_FIELDS,
    )
    decisions_count = Counter(str(row.get("decision", "")) for row in decisions)
    imported_by_label = Counter(str(row.get("label", "")) for row in import_rows)
    lines = [
        "SECOND TRAINING DATA IMPORT SUMMARY",
        "",
        f"Mode: {'COMMIT' if commit else 'DRY RUN'}",
        f"Candidates found: {candidate_count}",
        f"Accepted after duplicate audit: {accepted_count}",
        f"Unique decisions: {decisions_count['keep']}",
        f"Duplicates excluded: {decisions_count['duplicate']}",
        f"Already imported: {decisions_count['already_imported']}",
        f"Fingerprint errors: {decisions_count['error']}",
        f"Documents imported: {len(import_rows)}",
        f"Import/preprocessing failures: {len(failures)}",
        f"Failed Layout pages: {len(failed_pages)}",
        "New-document split policy: 100% train",
        "Existing validation/test rows modified: NO",
        f"Backup: {backup_root or 'not created in dry run'}",
        "",
        "Imported by label:",
    ]
    lines.extend(f"{label}: {imported_by_label[label]}" for label in sorted(imported_by_label))
    lines.extend(
        [
            "",
            f"Duplicate report: {RESULTS_DIR / 'second_training_duplicate_report.csv'}",
            f"Import manifest: {RESULTS_DIR / 'second_training_import_manifest.csv'}",
            f"Failures: {RESULTS_DIR / 'second_training_import_failures.csv'}",
            f"Failed pages: {RESULTS_DIR / 'second_training_failed_pages.csv'}",
            "Training was not started.",
        ]
    )
    (RESULTS_DIR / "second_training_import_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def print_plan(candidates, accepted, decisions, failures) -> None:
    print("\nSECOND TRAINING DRY RUN")
    print("=" * 72)
    print(f"Candidates: {len(candidates)}")
    print(f"Accepted as unique: {len(accepted)}")
    counts = Counter(row["decision"] for row in decisions)
    print(f"Duplicates: {counts['duplicate']}")
    print(f"Already imported: {counts['already_imported']}")
    print(f"Errors: {len(failures)}")
    by_label = Counter(row["label"] for row in accepted)
    for label in sorted(by_label):
        print(f"{label}: {by_label[label]} -> train")
    print("No files or dataset tables were changed. Re-run with --commit to import.")


def main() -> None:
    args = parse_args()
    if args.worker_job:
        worker_main(args.worker_job.resolve())
        return

    candidate_root = resolve_candidate_root(args.candidate_root)
    existing_metadata = read_csv_rows(METADATA_PATH, METADATA_FIELDS)
    candidates, inventory_failures = collect_candidates(candidate_root)
    if not candidates:
        raise ValueError(f"No supported candidates found under: {candidate_root}")
    accepted, decisions, audit_failures = audit_candidates(
        candidates,
        existing_metadata,
        image_threshold=args.image_similarity_threshold,
        text_threshold=args.text_similarity_threshold,
    )
    failures = inventory_failures + audit_failures

    if not args.commit:
        write_reports(
            decisions,
            failures,
            [],
            [],
            commit=False,
            backup_root=None,
            candidate_count=len(candidates),
            accepted_count=len(accepted),
        )
        print_plan(candidates, accepted, decisions, failures)
        return

    prepared, preprocessing_failures = preprocess_candidates(
        accepted, args.per_document_timeout
    )
    failures.extend(preprocessing_failures)
    import_rows, artifact_failures, backup_root, failed_pages, _ = commit_import(
        prepared, existing_metadata
    )
    failures.extend(artifact_failures)
    write_reports(
        decisions,
        failures,
        import_rows,
        failed_pages,
        commit=True,
        backup_root=backup_root,
        candidate_count=len(candidates),
        accepted_count=len(accepted),
    )
    print("\nSECOND TRAINING IMPORT COMPLETE")
    print("=" * 72)
    print(f"Imported documents: {len(import_rows)}")
    for label, count in sorted(Counter(row["label"] for row in import_rows).items()):
        print(f"{label}: {count} -> train")
    print(f"Failures: {len(failures)}")
    print(f"Backup: {backup_root}")
    print(f"Summary: {RESULTS_DIR / 'second_training_import_summary.txt'}")
    print("Training was NOT started.")


if __name__ == "__main__":
    main()
