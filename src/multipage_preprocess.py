from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import fitz
from PIL import Image

try:
    from .multipage import select_representative_pages
    from .preprocess import clean_text, render_pdf_page, run_ocr_on_image
except ImportError:
    from multipage import select_representative_pages  # type: ignore
    from preprocess import clean_text, render_pdf_page, run_ocr_on_image  # type: ignore


LAYOUT_STATUS_VALID = "valid"
LAYOUT_STATUS_EMPTY = "empty_or_failed_ocr"
LAYOUT_STATUS_PAGE_FAILED = "page_processing_failed"


@dataclass(slots=True)
class PageArtifact:
    document_id: str
    parent_document_id: str
    augmentation_group_id: str
    label: str
    split: str
    page_index: int
    total_pages: int
    image_path: Path
    ocr_path: Path
    words: list[str]
    boxes: list[list[int]]
    extraction_method: str
    layout_status: str = LAYOUT_STATUS_VALID
    failure_reason: str = ""

    @property
    def is_layout_valid(self) -> bool:
        return (
            self.layout_status == LAYOUT_STATUS_VALID
            and bool(self.words)
            and len(self.words) == len(self.boxes)
        )

    def manifest_row(self, project_root: Path) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "parent_document_id": self.parent_document_id,
            "augmentation_group_id": self.augmentation_group_id,
            "label": self.label,
            "split": self.split,
            "page_index": self.page_index,
            "total_pages": self.total_pages,
            "selected_page_count": 0,
            "image_path": relative_path(self.image_path, project_root),
            "ocr_path": relative_path(self.ocr_path, project_root),
            "extraction_method": self.extraction_method,
            "layout_status": self.layout_status,
            "failure_reason": self.failure_reason,
        }


PAGE_MANIFEST_FIELDS = (
    "document_id",
    "parent_document_id",
    "augmentation_group_id",
    "label",
    "split",
    "page_index",
    "total_pages",
    "selected_page_count",
    "image_path",
    "ocr_path",
    "extraction_method",
    "layout_status",
    "failure_reason",
)


def relative_path(path: Path, project_root: Path) -> str:
    path = Path(path).resolve()
    project_root = Path(project_root).resolve()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    image.convert("RGB").save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def _clean_pdf_words(page, image: Image.Image) -> tuple[list[str], list[list[int]]]:
    page_width = max(float(page.rect.width), 1.0)
    page_height = max(float(page.rect.height), 1.0)
    scale_x = image.width / page_width
    scale_y = image.height / page_height
    words: list[str] = []
    boxes: list[list[int]] = []
    for item in page.get_text("words", sort=True):
        if len(item) < 5:
            continue
        word = clean_text(item[4])
        if not word:
            continue
        x1 = max(0, min(image.width, int(round(float(item[0]) * scale_x))))
        y1 = max(0, min(image.height, int(round(float(item[1]) * scale_y))))
        x2 = max(x1, min(image.width, int(round(float(item[2]) * scale_x))))
        y2 = max(y1, min(image.height, int(round(float(item[3]) * scale_y))))
        words.append(word)
        boxes.append([x1, y1, x2, y2])
    return words, boxes


def extract_page_words_boxes(
    page, image: Image.Image, label: str, page_index: int
) -> tuple[list[str], list[list[int]], str, str, str]:
    embedded_error = ""
    try:
        words, boxes = _clean_pdf_words(page, image)
    except Exception as error:
        words, boxes = [], []
        embedded_error = f"Embedded PDF word extraction failed: {error}"
    if words and len(words) == len(boxes):
        return words, boxes, "pdf_embedded_words", LAYOUT_STATUS_VALID, ""

    try:
        _, payload = run_ocr_on_image(image, label, page_index=page_index)
        ocr_words = payload.get("words", [])
        ocr_boxes = payload.get("boxes", [])
        words = []
        boxes = []
        if isinstance(ocr_words, list) and isinstance(ocr_boxes, list):
            for word, box in zip(ocr_words, ocr_boxes):
                cleaned = clean_text(word)
                if not cleaned or not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = (int(value) for value in box)
                except (TypeError, ValueError):
                    continue
                x1 = max(0, min(image.width, x1))
                y1 = max(0, min(image.height, y1))
                x2 = max(x1, min(image.width, x2))
                y2 = max(y1, min(image.height, y2))
                words.append(cleaned)
                boxes.append([x1, y1, x2, y2])
        if words and len(words) == len(boxes):
            return words, boxes, "tesseract_ocr", LAYOUT_STATUS_VALID, ""
        reason = "OCR returned no aligned words and boxes"
        if embedded_error:
            reason = f"{embedded_error}; {reason}"
        return [], [], "tesseract_ocr_empty", LAYOUT_STATUS_EMPTY, reason
    except Exception as error:
        reason = f"OCR fallback failed: {error}"
        if embedded_error:
            reason = f"{embedded_error}; {reason}"
        return [], [], "ocr_failed", LAYOUT_STATUS_EMPTY, reason


def _page_payload(artifact: PageArtifact, image: Image.Image) -> dict[str, object]:
    return {
        "document_id": artifact.document_id,
        "parent_document_id": artifact.parent_document_id,
        "augmentation_group_id": artifact.augmentation_group_id,
        "page_index": artifact.page_index,
        "total_pages": artifact.total_pages,
        "label": artifact.label,
        "split": artifact.split,
        "image_path": str(artifact.image_path),
        "words": artifact.words,
        "boxes": artifact.boxes,
        "image_width": image.width,
        "image_height": image.height,
        "extraction_method": artifact.extraction_method,
        "layout_status": artifact.layout_status,
        "failure_reason": artifact.failure_reason,
    }


def _artifact_paths(output_root: Path, document_id: str, page_index: int) -> tuple[Path, Path]:
    stem = f"{document_id}__page_{page_index + 1:04d}"
    return output_root / "images" / f"{stem}.png", output_root / "ocr" / f"{stem}.json"


def _existing_artifact(
    image_path: Path,
    ocr_path: Path,
    document_row: Mapping[str, object],
    page_index: int,
    total_pages: int,
) -> PageArtifact | None:
    try:
        payload = json.loads(ocr_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("document_id") != document_row["document_id"]
        or int(payload.get("page_index", -1)) != page_index
    ):
        return None
    words = payload.get("words", [])
    boxes = payload.get("boxes", [])
    if not isinstance(words, list) or not isinstance(boxes, list):
        return None
    inferred_status = (
        LAYOUT_STATUS_VALID
        if words and len(words) == len(boxes)
        else LAYOUT_STATUS_EMPTY
    )
    layout_status = str(payload.get("layout_status", inferred_status))
    if layout_status == LAYOUT_STATUS_VALID and (not words or len(words) != len(boxes)):
        layout_status = LAYOUT_STATUS_EMPTY
    return PageArtifact(
        document_id=str(document_row["document_id"]),
        parent_document_id=str(document_row["parent_document_id"]),
        augmentation_group_id=str(document_row["augmentation_group_id"]),
        label=str(document_row["label"]),
        split=str(document_row["split"]),
        page_index=page_index,
        total_pages=total_pages,
        image_path=image_path,
        ocr_path=ocr_path,
        words=list(words),
        boxes=list(boxes),
        extraction_method=str(payload.get("extraction_method", "unknown")),
        layout_status=layout_status,
        failure_reason=str(payload.get("failure_reason", "")),
    )


def prepare_pdf_page_artifacts(
    pdf_path: Path,
    output_root: Path,
    document_row: Mapping[str, object],
    *,
    skip_existing: bool = False,
) -> tuple[int, list[int], list[PageArtifact]]:
    document = fitz.open(str(pdf_path))
    try:
        total_pages = document.page_count
        if total_pages < 1:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        selected_indices = select_representative_pages(total_pages)
        artifacts: list[PageArtifact] = []
        for page_index in selected_indices:
            image_path, ocr_path = _artifact_paths(
                Path(output_root), str(document_row["document_id"]), page_index
            )
            if skip_existing and image_path.is_file() and ocr_path.is_file():
                existing = _existing_artifact(
                    image_path, ocr_path, document_row, page_index, total_pages
                )
                if existing is not None:
                    artifacts.append(existing)
                    continue

            try:
                page = document.load_page(page_index)
                image = render_pdf_page(page).convert("RGB")
            except Exception as error:
                artifacts.append(
                    PageArtifact(
                        document_id=str(document_row["document_id"]),
                        parent_document_id=str(document_row["parent_document_id"]),
                        augmentation_group_id=str(document_row["augmentation_group_id"]),
                        label=str(document_row["label"]),
                        split=str(document_row["split"]),
                        page_index=page_index,
                        total_pages=total_pages,
                        image_path=image_path,
                        ocr_path=ocr_path,
                        words=[],
                        boxes=[],
                        extraction_method="render_failed",
                        layout_status=LAYOUT_STATUS_PAGE_FAILED,
                        failure_reason=f"Page render failed: {error}",
                    )
                )
                continue

            try:
                _atomic_save_image(image, image_path)
            except Exception as error:
                artifacts.append(
                    PageArtifact(
                        document_id=str(document_row["document_id"]),
                        parent_document_id=str(document_row["parent_document_id"]),
                        augmentation_group_id=str(document_row["augmentation_group_id"]),
                        label=str(document_row["label"]),
                        split=str(document_row["split"]),
                        page_index=page_index,
                        total_pages=total_pages,
                        image_path=image_path,
                        ocr_path=ocr_path,
                        words=[],
                        boxes=[],
                        extraction_method="image_save_failed",
                        layout_status=LAYOUT_STATUS_PAGE_FAILED,
                        failure_reason=f"Rendered image could not be saved: {error}",
                    )
                )
                continue

            words, boxes, method, layout_status, failure_reason = extract_page_words_boxes(
                page, image, str(document_row["label"]), page_index
            )
            artifact = PageArtifact(
                document_id=str(document_row["document_id"]),
                parent_document_id=str(document_row["parent_document_id"]),
                augmentation_group_id=str(document_row["augmentation_group_id"]),
                label=str(document_row["label"]),
                split=str(document_row["split"]),
                page_index=page_index,
                total_pages=total_pages,
                image_path=image_path,
                ocr_path=ocr_path,
                words=words,
                boxes=boxes,
                extraction_method=method,
                layout_status=layout_status,
                failure_reason=failure_reason,
            )
            try:
                _atomic_write_json(ocr_path, _page_payload(artifact, image))
            except Exception as error:
                artifact.layout_status = LAYOUT_STATUS_PAGE_FAILED
                artifact.failure_reason = f"OCR JSON could not be saved: {error}"
            artifacts.append(artifact)
        return total_pages, selected_indices, artifacts
    finally:
        document.close()


def prepare_single_image_artifact(
    source_image_path: Path,
    source_ocr_path: Path | None,
    output_root: Path,
    document_row: Mapping[str, object],
    *,
    skip_existing: bool = False,
) -> tuple[int, list[int], list[PageArtifact]]:
    image_path, ocr_path = _artifact_paths(Path(output_root), str(document_row["document_id"]), 0)
    if skip_existing and image_path.is_file() and ocr_path.is_file():
        existing = _existing_artifact(image_path, ocr_path, document_row, 0, 1)
        if existing is not None:
            return 1, [0], [existing]

    with Image.open(source_image_path) as source:
        image = source.convert("RGB")
    words: list[str] = []
    boxes: list[list[int]] = []
    method = "existing_ocr"
    layout_status = LAYOUT_STATUS_VALID
    failure_reason = ""
    if source_ocr_path and source_ocr_path.is_file():
        try:
            payload = json.loads(source_ocr_path.read_text(encoding="utf-8"))
            source_words = payload.get("words", [])
            source_boxes = payload.get("boxes", [])
            source_pages = payload.get("page_indices", [])
            if source_words and len(source_words) == len(source_boxes):
                for index, (word, box) in enumerate(zip(source_words, source_boxes)):
                    if source_pages and index < len(source_pages) and int(source_pages[index]) != 0:
                        continue
                    cleaned = clean_text(word)
                    if cleaned:
                        words.append(cleaned)
                        boxes.append([int(value) for value in box])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            words, boxes = [], []
    if not words:
        try:
            _, payload = run_ocr_on_image(image, str(document_row["label"]), page_index=0)
            words = []
            boxes = []
            for word, box in zip(payload.get("words", []), payload.get("boxes", [])):
                cleaned = clean_text(word)
                if cleaned and isinstance(box, (list, tuple)) and len(box) == 4:
                    words.append(cleaned)
                    boxes.append([int(value) for value in box])
            method = "tesseract_ocr"
        except Exception as error:
            words, boxes = [], []
            method = "ocr_failed"
            failure_reason = f"OCR fallback failed: {error}"
    if not words or len(words) != len(boxes):
        words, boxes = [], []
        layout_status = LAYOUT_STATUS_EMPTY
        if not failure_reason:
            failure_reason = "OCR returned no aligned words and boxes"

    artifact = PageArtifact(
        document_id=str(document_row["document_id"]),
        parent_document_id=str(document_row["parent_document_id"]),
        augmentation_group_id=str(document_row["augmentation_group_id"]),
        label=str(document_row["label"]),
        split=str(document_row["split"]),
        page_index=0,
        total_pages=1,
        image_path=image_path,
        ocr_path=ocr_path,
        words=words,
        boxes=boxes,
        extraction_method=method,
        layout_status=layout_status,
        failure_reason=failure_reason,
    )
    _atomic_save_image(image, image_path)
    _atomic_write_json(ocr_path, _page_payload(artifact, image))
    return 1, [0], [artifact]


def prepare_document_page_artifacts(
    project_root: Path,
    output_root: Path,
    document_row: Mapping[str, object],
    *,
    skip_existing: bool = False,
) -> tuple[int, list[int], list[PageArtifact]]:
    raw_path = resolve_path(project_root, str(document_row["raw_path"]))
    if raw_path.suffix.casefold() == ".pdf":
        return prepare_pdf_page_artifacts(
            raw_path, output_root, document_row, skip_existing=skip_existing
        )
    image_path = resolve_path(project_root, str(document_row.get("image_path", raw_path)))
    ocr_value = str(document_row.get("ocr_path", ""))
    ocr_path = resolve_path(project_root, ocr_value) if ocr_value else None
    return prepare_single_image_artifact(
        image_path,
        ocr_path,
        output_root,
        document_row,
        skip_existing=skip_existing,
    )


def prepare_inference_file_artifacts(
    document_path: Path,
    output_root: Path,
    *,
    source_image_path: Path | None = None,
    source_ocr_path: Path | None = None,
) -> tuple[int, list[int], list[PageArtifact]]:
    """Create the same aligned page artifacts used by training for one live document."""
    document_path = Path(document_path)
    document_row = {
        "document_id": "inference_document",
        "parent_document_id": "inference_document",
        "augmentation_group_id": "inference_document",
        "label": "unknown",
        "split": "inference",
    }
    if document_path.suffix.casefold() == ".pdf":
        return prepare_pdf_page_artifacts(document_path, output_root, document_row)
    image_path = Path(source_image_path or document_path)
    return prepare_single_image_artifact(
        image_path,
        Path(source_ocr_path) if source_ocr_path else None,
        output_root,
        document_row,
    )
