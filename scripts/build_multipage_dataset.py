from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    atomic_write_csv,
    build_document_manifest,
    read_csv,
    write_document_manifest,
)
from src.multipage_preprocess import (  # noqa: E402
    LAYOUT_STATUS_EMPTY,
    LAYOUT_STATUS_VALID,
    PAGE_MANIFEST_FIELDS,
    prepare_document_page_artifacts,
)


DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "metadata.csv"
SOURCE_TRACKING_PATH = DATA_DIR / "dataset_sources_extra.csv"
MULTIPAGE_DIR = DATA_DIR / "multipage"
DOCUMENT_MANIFEST_PATH = MULTIPAGE_DIR / "document_manifest.csv"
TOKENIZER_DIR = PROJECT_ROOT / "models" / "xlm_roberta" / "best_model"
RESULTS_DIR = PROJECT_ROOT / "results" / "dataset_expansion"

FAILED_PAGE_FIELDS = (
    "document_id",
    "parent_document_id",
    "augmentation_group_id",
    "label",
    "split",
    "page_index",
    "page_number",
    "raw_path",
    "status",
    "reason",
    "error_message",
    "image_path",
)

FAILED_DOCUMENT_FIELDS = (
    "document_id",
    "parent_document_id",
    "augmentation_group_id",
    "label",
    "split",
    "raw_path",
    "status",
    "reason",
    "error_message",
    "image_page_count",
    "valid_layout_page_count",
    "chunk_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe multi-page dataset artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-raw-hashes", action="store_true")
    parser.add_argument("--max-documents-per-class-per-split", type=int)
    return parser.parse_args()


def load_or_build_manifest(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.skip_existing and DOCUMENT_MANIFEST_PATH.is_file():
        rows = read_csv(DOCUMENT_MANIFEST_PATH)
        validate_document_manifest(rows)
        print(f"Using existing document manifest: {DOCUMENT_MANIFEST_PATH}")
        return rows

    metadata = read_csv(METADATA_PATH)
    sources = read_csv(SOURCE_TRACKING_PATH) if SOURCE_TRACKING_PATH.is_file() else []
    print(f"Building group-aware document manifest for {len(metadata)} documents...")
    rows = build_document_manifest(
        PROJECT_ROOT,
        metadata,
        sources,
        seed=args.seed,
        compute_hashes=not args.no_raw_hashes,
    )
    write_document_manifest(DOCUMENT_MANIFEST_PATH, rows)
    print(f"Document manifest: {DOCUMENT_MANIFEST_PATH}")
    return rows


def limited_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    limit = args.max_documents_per_class_per_split
    if args.smoke_test and limit is None:
        limit = 1
    if limit is None:
        return rows
    selected: list[dict[str, str]] = []
    counts = Counter()
    for row in rows:
        key = (row["split"], row["label"])
        if counts[key] >= limit:
            continue
        selected.append(row)
        counts[key] += 1
    return selected


def atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def document_failure(
    row: dict[str, str],
    *,
    status: str,
    reason: str,
    error_message: str = "",
    image_page_count: int = 0,
    valid_layout_page_count: int = 0,
    chunk_count: int = 0,
) -> dict[str, object]:
    return {
        "document_id": row["document_id"],
        "parent_document_id": row["parent_document_id"],
        "augmentation_group_id": row["augmentation_group_id"],
        "label": row["label"],
        "split": row["split"],
        "raw_path": row["raw_path"],
        "status": status,
        "reason": reason,
        "error_message": error_message,
        "image_page_count": image_page_count,
        "valid_layout_page_count": valid_layout_page_count,
        "chunk_count": chunk_count,
    }


def build_page_artifacts(
    rows: list[dict[str, str]], output_root: Path, skip_existing: bool
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    Counter,
]:
    page_rows: list[dict[str, object]] = []
    failed_pages: list[dict[str, object]] = []
    failed_documents: list[dict[str, object]] = []
    stats = Counter(
        documents_total=len(rows),
        documents_with_valid_layout_page=0,
        documents_without_valid_layout_page=0,
        skipped_empty_pages=0,
        failed_page_processing=0,
    )
    for row in tqdm(rows, desc="Multi-page image/OCR preprocessing"):
        try:
            total_pages, selected_indices, artifacts = prepare_document_page_artifacts(
                PROJECT_ROOT,
                output_root / "pages",
                row,
                skip_existing=skip_existing,
            )
        except Exception as error:
            failed_documents.append(
                document_failure(
                    row,
                    status="document_processing_failed",
                    reason="Document page preprocessing raised an exception",
                    error_message=f"{type(error).__name__}: {error}",
                )
            )
            stats["documents_without_valid_layout_page"] += 1
            tqdm.write(f"SKIPPED document: {row['raw_path']} ({error})")
            continue

        image_page_count = 0
        valid_layout_page_count = 0
        for artifact in artifacts:
            image_exists = artifact.image_path.is_file()
            ocr_exists = artifact.ocr_path.is_file()
            if image_exists:
                manifest_row = artifact.manifest_row(PROJECT_ROOT)
                manifest_row["total_pages"] = total_pages
                manifest_row["selected_page_count"] = len(selected_indices)
                page_rows.append(manifest_row)
                image_page_count += 1

            if artifact.is_layout_valid and image_exists and ocr_exists:
                valid_layout_page_count += 1
                continue

            if artifact.layout_status == LAYOUT_STATUS_EMPTY:
                stats["skipped_empty_pages"] += 1
            else:
                stats["failed_page_processing"] += 1
            failed_pages.append(
                {
                    "document_id": row["document_id"],
                    "parent_document_id": row["parent_document_id"],
                    "augmentation_group_id": row["augmentation_group_id"],
                    "label": row["label"],
                    "split": row["split"],
                    "page_index": artifact.page_index,
                    "page_number": artifact.page_index + 1,
                    "raw_path": row["raw_path"],
                    "status": artifact.layout_status,
                    "reason": artifact.failure_reason or "Page is not valid for LayoutLMv3",
                    "error_message": artifact.failure_reason,
                    "image_path": (
                        artifact.manifest_row(PROJECT_ROOT)["image_path"] if image_exists else ""
                    ),
                }
            )
            tqdm.write(
                f"SKIPPED Layout page {artifact.page_index + 1}: {row['raw_path']} "
                f"({artifact.layout_status})"
            )

        if valid_layout_page_count:
            stats["documents_with_valid_layout_page"] += 1
        else:
            stats["documents_without_valid_layout_page"] += 1
            failed_documents.append(
                document_failure(
                    row,
                    status="failed_for_layoutlm",
                    reason="Document has no valid OCR page artifacts",
                    image_page_count=image_page_count,
                    valid_layout_page_count=0,
                )
            )
    validate_artifact_rows(rows, page_rows, artifact_name="page manifest")
    atomic_write_csv(output_root / "page_manifest.csv", page_rows, PAGE_MANIFEST_FIELDS)
    return page_rows, failed_pages, failed_documents, stats


def build_chunk_artifacts(
    rows: list[dict[str, str]], output_root: Path, skip_existing: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not TOKENIZER_DIR.is_dir():
        raise FileNotFoundError(f"Tokenizer is missing: {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    existing_by_document: dict[str, list[dict[str, object]]] = {}
    chunk_path = output_root / "chunk_manifest.jsonl"
    if skip_existing and chunk_path.is_file():
        for chunk in read_jsonl(chunk_path):
            existing_by_document.setdefault(str(chunk.get("document_id", "")), []).append(chunk)

    chunk_rows: list[dict[str, object]] = []
    failed_documents: list[dict[str, object]] = []
    for row in tqdm(rows, desc="XLM-R overflow chunking"):
        existing = existing_by_document.get(row["document_id"], [])
        if existing and all(
            str(chunk.get(field, "")) == str(row[field])
            for chunk in existing
            for field in ("parent_document_id", "augmentation_group_id", "label", "split")
        ):
            chunk_rows.extend(existing)
            continue

        try:
            text_path = PROJECT_ROOT / Path(row["text_path"])
            text = text_path.read_text(encoding="utf-8", errors="ignore")
            chunks = tokenize_document_chunks(
                tokenizer,
                text,
                max_length=CHUNK_MAX_LENGTH,
                stride=CHUNK_STRIDE,
                max_chunks=MAX_SELECTED_CHUNKS,
            )
        except Exception as error:
            failed_documents.append(
                document_failure(
                    row,
                    status="failed_for_xlm_roberta",
                    reason="Text chunk generation failed",
                    error_message=f"{type(error).__name__}: {error}",
                )
            )
            tqdm.write(f"SKIPPED text chunks: {row['raw_path']} ({error})")
            continue

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
    validate_artifact_rows(rows, chunk_rows, artifact_name="chunk manifest")
    atomic_write_jsonl(chunk_path, chunk_rows)
    return chunk_rows, failed_documents


def write_failure_reports(
    report_root: Path,
    failed_pages: list[dict[str, object]],
    failed_documents: list[dict[str, object]],
) -> tuple[Path, Path]:
    page_report = report_root / "multipage_failed_pages.csv"
    document_report = report_root / "multipage_failed_documents.csv"
    atomic_write_csv(page_report, failed_pages, FAILED_PAGE_FIELDS)
    atomic_write_csv(document_report, failed_documents, FAILED_DOCUMENT_FIELDS)
    return page_report, document_report


def write_summary(
    report_root: Path,
    *,
    selected_documents: int,
    page_rows: list[dict[str, object]],
    chunk_rows: list[dict[str, object]],
    page_stats: Counter,
    failed_documents: list[dict[str, object]],
    page_report: Path,
    document_report: Path,
) -> Path:
    summary_path = report_root / "multipage_preprocessing_summary.txt"
    report_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "MULTI-PAGE PREPROCESSING SUMMARY",
        "",
        f"Documents selected: {selected_documents}",
        f"Page image artifacts: {len(page_rows)}",
        f"XLM-RoBERTa chunks: {len(chunk_rows)}",
        f"Skipped empty/OCR-failed pages: {page_stats['skipped_empty_pages']}",
        f"Other failed pages: {page_stats['failed_page_processing']}",
        (
            "Documents with at least one valid LayoutLMv3 page: "
            f"{page_stats['documents_with_valid_layout_page']}"
        ),
        (
            "Documents without a valid OCR/page artifact: "
            f"{page_stats['documents_without_valid_layout_page']}"
        ),
        f"Document failure records: {len(failed_documents)}",
        "Pipeline continued after individual page/document failures: YES",
        "",
        f"Failed page report: {page_report}",
        f"Failed document report: {document_report}",
    ]
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    return summary_path


def print_manifest_summary(rows: list[dict[str, str]]) -> None:
    print("DOCUMENT SPLITS")
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        labels = Counter(row["label"] for row in split_rows)
        print(f"{split}: {len(split_rows)} | {dict(labels)}")


def main() -> None:
    args = parse_args()
    document_rows = load_or_build_manifest(args)
    validate_document_manifest(document_rows)
    print_manifest_summary(document_rows)
    if args.manifest_only:
        print("Manifest-only mode: page and chunk artifacts were not generated.")
        return

    selected = limited_rows(document_rows, args)
    output_root = MULTIPAGE_DIR / "smoke_test" if args.smoke_test else MULTIPAGE_DIR
    report_root = RESULTS_DIR / "multipage_smoke_test" if args.smoke_test else RESULTS_DIR
    page_rows, failed_pages, failed_documents, page_stats = build_page_artifacts(
        selected, output_root, args.skip_existing
    )
    chunk_rows, chunk_failures = build_chunk_artifacts(
        selected, output_root, args.skip_existing
    )
    failed_documents.extend(chunk_failures)
    page_report, document_report = write_failure_reports(
        report_root, failed_pages, failed_documents
    )
    summary_path = write_summary(
        report_root,
        selected_documents=len(selected),
        page_rows=page_rows,
        chunk_rows=chunk_rows,
        page_stats=page_stats,
        failed_documents=failed_documents,
        page_report=page_report,
        document_report=document_report,
    )
    print()
    print(f"Documents processed: {len(selected)}")
    print(f"Page artifacts: {len(page_rows)}")
    print(f"Chunk artifacts: {len(chunk_rows)}")
    print(f"Skipped empty/OCR-failed pages: {page_stats['skipped_empty_pages']}")
    print(
        "Documents with at least one valid LayoutLMv3 page: "
        f"{page_stats['documents_with_valid_layout_page']}"
    )
    print(
        "Documents without valid LayoutLMv3 pages: "
        f"{page_stats['documents_without_valid_layout_page']}"
    )
    print("Pipeline continued after individual page/document failures: YES")
    print(f"Page manifest: {output_root / 'page_manifest.csv'}")
    print(f"Chunk manifest: {output_root / 'chunk_manifest.jsonl'}")
    print(f"Failed page report: {page_report}")
    print(f"Failed document report: {document_report}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
