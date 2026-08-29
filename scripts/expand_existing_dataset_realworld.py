"""Safely expand the existing dataset with diverse, traceable documents.

Dry-run mode is project read-only. A real run downloads candidates, rejects
duplicates, preprocesses only accepted additions, appends metadata, and creates
group-aware 75/15/10 splits.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import requests
import fitz
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_expansion import (  # noqa: E402
    CLASS_NAMES,
    METADATA_FIELDS,
    SUPPORTED_EXTENSIONS,
    DatasetExpansionError,
    bit_similarity,
    atomic_write_csv,
    atomic_write_text,
    build_fingerprint_record,
    class_counts,
    document_preview,
    group_aware_stratified_split,
    load_existing_fingerprint_index,
    normalize_text,
    perceptual_hash,
    project_path,
    read_csv_rows,
    relative_project_path,
    sha256_file,
    validate_training_path,
    visible_html_text,
)
from src.preprocess import TESSERACT_AVAILABLE, TESSERACT_COMMAND, process_file_to_outputs  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_PATH = DATA_DIR / "metadata.csv"
SPLITS_DIR = DATA_DIR / "splits"
SOURCE_TRACKING_PATH = DATA_DIR / "dataset_sources_extra.csv"
METADATA_BACKUP_PATH = DATA_DIR / "metadata_before_expansion.csv"
SPLITS_BACKUP_DIR = DATA_DIR / "splits_before_expansion"
RESULTS_DIR = PROJECT_ROOT / "results" / "dataset_expansion"
SOURCE_CONFIG_PATH = PROJECT_ROOT / "config" / "dataset_expansion_sources.json"
PREPROCESSING_FAILURE_PATH = RESULTS_DIR / "preprocessing_failed_documents.csv"
STAGING_MANIFEST_NAME = "staging_manifest.csv"
HF_SHUFFLE_BUFFER_MAX = 4
HF_ROWS_API_URL = "https://datasets-server.huggingface.co/rows"

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
    "source",
    "decision",
    "reason",
    "similar_to",
    "similarity_score",
)
FAILURE_FIELDS = (
    "id",
    "label",
    "raw_path",
    "source",
    "reason",
    "error_message",
    "elapsed_seconds",
)
STAGING_MANIFEST_FIELDS = (
    "id",
    "label",
    "source_name",
    "source_locator",
    "original_id",
    "raw_file",
    "raw_final_path",
    "text_hint_file",
    "is_augmented",
    "augmentation_type",
    "parent_id",
)

IMAGE_FIELD_HINTS = (
    "image",
    "image_base64",
    "page_image",
    "document_image",
    "scan",
    "jpg",
    "png",
)
FILE_FIELD_HINTS = (
    "pdf_bytes_base64",
    "pdf_bytes",
    "file_bytes",
    "file_content",
    "document",
    "pdf",
    "file",
)
TEXT_FIELD_HINTS = (
    "text",
    "article",
    "contract",
    "contract_text",
    "document_text",
    "email_body",
    "body",
    "content",
    "raw",
    "abstract",
)
ID_FIELD_HINTS = ("id", "document_id", "file_name", "filename", "name", "arxiv_id")
LABEL_FIELD_HINTS = ("class_name", "label", "class", "category", "document_type")


@dataclass(slots=True)
class CandidateAsset:
    label: str
    source_name: str
    source_locator: str
    original_id: str
    payload: bytes
    extension: str
    text: str = ""


@dataclass(slots=True)
class AcceptedDocument:
    document_id: str
    label: str
    source_name: str
    source_locator: str
    original_id: str
    raw_staging_path: Path
    raw_final_path: Path
    text_hint: str
    is_augmented: bool = False
    augmentation_type: str = ""
    parent_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand data/raw with diverse documents while preventing duplicate leakage."
    )
    parser.add_argument("--target-per-class", type=int, default=750)
    parser.add_argument("--dry-run", action="store_true", help="Print a read-only plan; do not download or write.")
    parser.add_argument("--augmentation-fraction", type=float, default=0.20)
    parser.add_argument("--max-source-share", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=int, default=90)
    parser.add_argument(
        "--per-document-timeout",
        type=int,
        default=120,
        help="Maximum preprocessing time in seconds for one document.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the newest existing expansion staging directory.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse complete processed outputs already present in staging or final output folders.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry IDs already listed in preprocessing_failed_documents.csv.",
    )
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG_PATH)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Use only a named configured source. May be supplied more than once.",
    )
    parser.add_argument("--image-similarity-threshold", type=float, default=0.94)
    parser.add_argument("--text-similarity-threshold", type=float, default=0.95)
    parser.add_argument("--keep-staging", action="store_true", help="Keep staging files after a failed real run.")
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.target_per_class < 1:
        parser.error("--target-per-class must be positive")
    if not 0 <= args.augmentation_fraction <= 0.30:
        parser.error("--augmentation-fraction must be between 0 and 0.30")
    if not 0 < args.max_source_share <= 1:
        parser.error("--max-source-share must be in (0, 1]")
    if args.per_document_timeout < 1:
        parser.error("--per-document-timeout must be at least 1 second")
    return args


def load_source_catalog(path: Path, selected: set[str]) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        raise FileNotFoundError(f"Source catalog is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise DatasetExpansionError(f"Invalid source catalog: {path}")

    names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise DatasetExpansionError("Every source entry must be an object.")
        missing = [key for key in ("name", "label", "adapter", "materialize", "url") if not source.get(key)]
        if missing:
            raise DatasetExpansionError(f"Source entry is missing {missing}: {source}")
        if source["name"] in names:
            raise DatasetExpansionError(f"Duplicate source name: {source['name']}")
        if source["label"] not in CLASS_NAMES:
            raise DatasetExpansionError(f"Unknown source label: {source['label']}")
        names.add(source["name"])
        if source.get("enabled", True) and (not selected or source["name"] in selected):
            validated.append(source)

    unknown = selected - names
    if unknown:
        raise DatasetExpansionError(f"Unknown --source value(s): {', '.join(sorted(unknown))}")
    return validated, str(payload.get("notes", ""))


def validate_metadata(rows: list[dict[str, str]]) -> None:
    seen_ids: set[str] = set()
    seen_raw: set[str] = set()
    for row in rows:
        label = row.get("label", "")
        if label not in CLASS_NAMES:
            raise DatasetExpansionError(f"Unexpected metadata label: {label!r}")
        if row["id"] in seen_ids:
            raise DatasetExpansionError(f"Duplicate metadata id: {row['id']}")
        if row["raw_path"] in seen_raw:
            raise DatasetExpansionError(f"Duplicate metadata raw_path: {row['raw_path']}")
        seen_ids.add(row["id"])
        seen_raw.add(row["raw_path"])
        raw_path = project_path(PROJECT_ROOT, row["raw_path"])
        validate_training_path(PROJECT_ROOT, raw_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Metadata raw_path does not exist: {raw_path}")


def allocate_quotas(
    sources: list[dict[str, Any]], total: int, max_source_share: float
) -> dict[str, int]:
    if total <= 0:
        return {source["name"]: 0 for source in sources}
    if not sources:
        return {}

    share_cap = max(1, math.ceil(total * max_source_share))
    capacities = {
        source["name"]: min(int(source.get("max_documents", total)), share_cap)
        for source in sources
    }
    if sum(capacities.values()) < total:
        raise DatasetExpansionError(
            f"Configured source capacity ({sum(capacities.values())}) is below required real documents ({total})."
        )

    quotas = {source["name"]: 0 for source in sources}
    weights = {source["name"]: max(float(source.get("weight", 1.0)), 0.01) for source in sources}
    for _ in range(total):
        available = [source["name"] for source in sources if quotas[source["name"]] < capacities[source["name"]]]
        chosen = min(available, key=lambda name: (quotas[name] / weights[name], quotas[name], name))
        quotas[chosen] += 1
    return quotas


def build_plan(
    current: Mapping[str, int],
    sources: list[dict[str, Any]],
    target: int,
    augmentation_fraction: float,
    max_source_share: float,
) -> dict[str, dict[str, Any]]:
    plan: dict[str, dict[str, Any]] = {}
    for label in CLASS_NAMES:
        additions = max(0, target - current.get(label, 0))
        augmentation_target = int(additions * augmentation_fraction)
        real_target = additions - augmentation_target
        label_sources = [source for source in sources if source["label"] == label]
        quotas = allocate_quotas(label_sources, real_target, max_source_share) if real_target else {}
        plan[label] = {
            "old_count": current.get(label, 0),
            "additions": additions,
            "real_target": real_target,
            "augmentation_target": augmentation_target,
            "quotas": quotas,
        }
    return plan


def print_plan(plan: Mapping[str, Mapping[str, Any]], sources: list[dict[str, Any]], dry_run: bool) -> None:
    by_name = {source["name"]: source for source in sources}
    print("\nDATASET EXPANSION PLAN")
    print("=" * 88)
    print(f"{'Class':<12}{'Existing':>10}{'Real new':>12}{'Augmented':>12}{'Target':>10}")
    for label in CLASS_NAMES:
        item = plan[label]
        target = item["old_count"] + item["additions"]
        print(
            f"{label:<12}{item['old_count']:>10}{item['real_target']:>12}"
            f"{item['augmentation_target']:>12}{target:>10}"
        )
        for name, quota in item["quotas"].items():
            source = by_name[name]
            print(f"  - {name}: {quota} planned ({source.get('license', 'license not listed')})")
    if dry_run:
        print("\nKnown similarity skips: 0")
        print(
            "Remote candidates are intentionally not downloaded in read-only dry-run; "
            "their duplicate count is determined before acceptance during the real run."
        )
        print("DRY RUN: no files, caches, metadata, splits, or reports were written.")


def decode_label(dataset: Any, row: Mapping[str, Any]) -> str:
    for field in LABEL_FIELD_HINTS:
        if field not in row:
            continue
        value = row[field]
        try:
            feature = dataset.features.get(field) if dataset.features else None
            if feature is not None and hasattr(feature, "int2str") and isinstance(value, (int, np.integer)):
                value = feature.int2str(int(value))
        except Exception:
            pass
        return str(value).strip().casefold().replace(" ", "_")
    return ""


def first_string(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def extract_row_text(row: Mapping[str, Any]) -> str:
    headers = []
    header_map = (
        ("Subject", ("subject", "subject_line", "title")),
        ("From", ("from", "sender", "from_address")),
        ("To", ("to", "recipient", "recipients", "to_address")),
        ("Date", ("date", "sent_at", "timestamp")),
    )
    for display, fields in header_map:
        value = first_string(row, fields)
        if value:
            headers.append(f"{display}: {value}")

    candidates: list[str] = []
    for field in TEXT_FIELD_HINTS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            if "base64" in field or (len(value) > 500 and re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", value)):
                continue
            candidates.append(value.strip())
    body = max(candidates, key=len, default="")
    file_content = row.get("file_content")
    if isinstance(file_content, str) and file_content.strip().startswith("<"):
        visible_file_content = visible_html_text(file_content)
        if len(visible_file_content) > len(body):
            body = visible_file_content
    if body.lstrip().startswith("<"):
        body = visible_html_text(body)
    return "\n".join(headers + (["", body] if body else [])).strip()


def detect_extension(payload: bytes, fallback: str = ".bin") -> str:
    if payload.startswith(b"%PDF"):
        return ".pdf"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload[:4] in {b"II*\x00", b"MM\x00*"}:
        return ".tif"
    return fallback.lower() if fallback.startswith(".") else f".{fallback.lower()}"


def image_to_bytes(image: Image.Image, format_name: str = "PNG") -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(output, format=format_name)
    return output.getvalue()


def decode_possible_base64(value: str) -> bytes | None:
    cleaned = value.strip()
    if cleaned.startswith("data:") and "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]
    try:
        payload = base64.b64decode(cleaned, validate=False)
    except Exception:
        return None
    return payload if len(payload) >= 64 else None


def extract_row_payload(row: Mapping[str, Any]) -> tuple[bytes | None, str]:
    for field in (*IMAGE_FIELD_HINTS, *FILE_FIELD_HINTS):
        if field not in row:
            continue
        value = row[field]
        if isinstance(value, Image.Image):
            return image_to_bytes(value), ".png"
        if isinstance(value, dict):
            raw_bytes = value.get("bytes")
            if isinstance(raw_bytes, bytes):
                return raw_bytes, detect_extension(raw_bytes, Path(str(value.get("path", ""))).suffix or ".bin")
            nested_path = value.get("path")
            if nested_path and Path(str(nested_path)).exists():
                payload = Path(str(nested_path)).read_bytes()
                return payload, detect_extension(payload, Path(str(nested_path)).suffix)
        if isinstance(value, bytes):
            return value, detect_extension(value)
        if isinstance(value, str) and ("base64" in field or "bytes" in field or field == "file_content"):
            looks_encoded = "base64" in field or "bytes" in field
            looks_encoded = looks_encoded or bool(
                len(value) > 128 and re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", value.strip())
            )
            if looks_encoded:
                payload = decode_possible_base64(value)
                if payload:
                    return payload, detect_extension(payload)
    return None, ""


def safe_original_id(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return value[:160] or fallback


def render_legal_pdf(text: str, title: str) -> bytes:
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=A4)
    width, height = A4
    margin = 55
    y = height - margin
    page_number = 1

    def start_page() -> None:
        nonlocal y, page_number
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(margin, height - 42, title[:80])
        pdf.setFont("Helvetica", 8)
        pdf.drawRightString(width - margin, 25, f"Page {page_number}")
        y = height - 68

    start_page()
    pdf.setFont("Times-Roman", 9.5)
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        lines = textwrap.wrap(re.sub(r"\s+", " ", paragraph), width=104) or [""]
        for line in lines:
            if y < 55:
                pdf.showPage()
                page_number += 1
                start_page()
                pdf.setFont("Times-Roman", 9.5)
            pdf.drawString(margin, y, line[:120])
            y -= 11.5
        y -= 5
    pdf.save()
    return output.getvalue()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    max_width: int,
    max_y: int,
    fill: str = "#202124",
    spacing: int = 8,
) -> int:
    x, y = xy
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        lines: list[str] = []
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if not lines:
            lines = [""]
        for line in lines:
            line_height = draw.textbbox((0, 0), line or "Ag", font=font)[3] + spacing
            if y + line_height > max_y:
                return y
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
        y += spacing
    return y


def render_webmail_png(text: str, variant_seed: str) -> bytes:
    rng = random.Random(variant_seed)
    width = rng.choice((1180, 1240, 1320))
    height = rng.choice((1480, 1580, 1680))
    image = Image.new("RGB", (width, height), "#f6f8fc")
    draw = ImageDraw.Draw(image)
    regular = load_font(25)
    small = load_font(20)
    bold = load_font(28, bold=True)

    draw.rectangle((0, 0, width, 78), fill="#ffffff")
    draw.text((34, 23), "Mail", font=bold, fill="#1f1f1f")
    draw.rounded_rectangle((160, 14, width - 36, 64), radius=22, fill="#edf2fa")
    draw.text((190, 25), "Search mail", font=small, fill="#5f6368")
    card_left, card_top = 42, 112
    card_right, card_bottom = width - 42, height - 40
    draw.rounded_rectangle((card_left, card_top, card_right, card_bottom), radius=12, fill="#ffffff")

    fields: dict[str, str] = {}
    body_lines: list[str] = []
    in_headers = True
    for line in text.splitlines():
        match = re.match(r"^(Subject|From|To|Date):\s*(.*)$", line, flags=re.IGNORECASE)
        if in_headers and match:
            fields[match.group(1).title()] = match.group(2).strip()
        elif not line.strip() and in_headers:
            in_headers = False
        else:
            body_lines.append(line)

    subject = fields.get("Subject") or "Message"
    sender = fields.get("From") or "sender@example.com"
    recipient = fields.get("To") or "recipient@example.com"
    date = fields.get("Date") or ""
    body = "\n".join(body_lines).strip() or text

    y = card_top + 34
    y = draw_wrapped(draw, subject, (card_left + 36, y), bold, card_right - card_left - 72, y + 120)
    draw.text((card_left + 36, y + 8), sender, font=regular, fill="#202124")
    if date:
        date_width = draw.textbbox((0, 0), date, font=small)[2]
        draw.text((card_right - 36 - date_width, y + 11), date, font=small, fill="#5f6368")
    y += 42
    draw.text((card_left + 36, y), f"to {recipient}", font=small, fill="#5f6368")
    y += 54
    draw.line((card_left + 30, y, card_right - 30, y), fill="#e0e0e0", width=2)
    y += 32
    draw_wrapped(
        draw,
        body,
        (card_left + 40, y),
        regular,
        card_right - card_left - 80,
        card_bottom - 40,
        spacing=10,
    )
    return image_to_bytes(image, "PNG")


def materialize_candidate(
    *,
    label: str,
    source: Mapping[str, Any],
    original_id: str,
    payload: bytes | None,
    extension: str,
    text: str,
) -> CandidateAsset | None:
    mode = source.get("materialize", "original")
    source_locator = str(source.get("url") or source.get("dataset") or "")
    if mode == "webmail_png":
        if len(normalize_text(text)) < 40:
            return None
        payload = render_webmail_png(text, f"{source['name']}:{original_id}")
        extension = ".png"
    elif mode == "legal_pdf" or not payload:
        if len(normalize_text(text)) < 200:
            return None
        payload = render_legal_pdf(text, f"Contract {original_id}")
        extension = ".pdf"

    if not payload or len(payload) < 128:
        return None
    extension = detect_extension(payload, extension or ".bin")
    if extension not in {".pdf", ".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                payload = image_to_bytes(image, "PNG")
            extension = ".png"
        except Exception:
            if len(normalize_text(text)) < 200:
                return None
            payload = render_legal_pdf(text, f"Document {original_id}")
            extension = ".pdf"
    return CandidateAsset(
        label=label,
        source_name=str(source["name"]),
        source_locator=source_locator,
        original_id=original_id,
        payload=payload,
        extension=extension,
        text=text,
    )


def streaming_shuffle_buffer_size(scan_limit: int) -> int:
    """Keep streaming image datasets bounded instead of buffering many GB in RAM."""
    return min(max(scan_limit, 1), HF_SHUFFLE_BUFFER_MAX)


def iter_huggingface_rows(source: Mapping[str, Any], scan_limit: int, seed: int) -> Iterator[CandidateAsset]:
    try:
        from datasets import Image as DatasetImage
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Missing dependency 'datasets'. Run: python -m pip install datasets") from error

    kwargs: dict[str, Any] = {
        "path": source["dataset"],
        "split": source.get("split", "train"),
        "streaming": True,
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    dataset = load_dataset(**kwargs)
    for field_name, feature in (getattr(dataset, "features", None) or {}).items():
        if isinstance(feature, DatasetImage):
            try:
                dataset = dataset.cast_column(field_name, DatasetImage(decode=False))
            except Exception as error:
                print(f"WARNING: could not disable Hugging Face image decoding for {field_name}: {error}")
    try:
        dataset = dataset.shuffle(seed=seed, buffer_size=streaming_shuffle_buffer_size(scan_limit))
    except Exception:
        pass

    accepted_labels = {
        str(value).casefold().replace(" ", "_") for value in source.get("accepted_labels", [])
    }
    scanned = 0
    for row_number, row in enumerate(dataset):
        if scanned >= scan_limit:
            break
        scanned += 1
        if accepted_labels and decode_label(dataset, row) not in accepted_labels:
            continue

        payload, extension = extract_row_payload(row)
        text = extract_row_text(row)
        raw_id = first_string(row, ID_FIELD_HINTS)
        fallback = f"row_{row_number:08d}"
        original_id = safe_original_id(raw_id, fallback)
        candidate = materialize_candidate(
            label=str(source["label"]),
            source=source,
            original_id=original_id,
            payload=payload,
            extension=extension,
            text=text,
        )
        if candidate:
            yield candidate


def iter_huggingface_rows_api(
    source: Mapping[str, Any], scan_limit: int, seed: int, timeout: int
) -> Iterator[CandidateAsset]:
    """Read bounded pages through Dataset Viewer instead of loading a multi-GB Parquet shard."""
    base_params = {
        "dataset": str(source["dataset"]),
        "config": str(source.get("config", "default")),
        "split": str(source.get("split", "train")),
    }
    headers = {"User-Agent": "document-ai-classifier-dataset-expansion/1.0"}
    retries = max(1, int(source.get("rows_api_retries", 5)))
    probe_data = request_huggingface_rows_page(
        {**base_params, "offset": 0, "length": 1}, headers, timeout, retries
    )
    total_rows = int(probe_data.get("num_rows_total", 0))
    if total_rows < 1:
        raise DatasetExpansionError(
            f"Hugging Face Rows API returned no rows for {source['dataset']}"
        )

    page_size = min(max(int(source.get("rows_api_page_size", 50)), 1), 100)
    page_offsets = list(range(0, total_rows, page_size))
    random.Random(seed).shuffle(page_offsets)
    accepted_labels = {
        str(value).casefold().replace(" ", "_") for value in source.get("accepted_labels", [])
    }
    scanned = 0

    for offset in page_offsets:
        if scanned >= scan_limit:
            return
        length = min(page_size, total_rows - offset, scan_limit - scanned)
        page_data = request_huggingface_rows_page(
            {**base_params, "offset": offset, "length": length},
            headers,
            timeout,
            retries,
        )
        entries = page_data.get("rows", [])
        if not entries:
            continue

        for entry in entries:
            if scanned >= scan_limit:
                return
            scanned += 1
            row = entry.get("row", {})
            if not isinstance(row, Mapping):
                continue
            decoded_label = first_string(row, LABEL_FIELD_HINTS).casefold().replace(" ", "_")
            if accepted_labels and decoded_label not in accepted_labels:
                continue

            row_index = int(entry.get("row_idx", offset))
            payload, extension = extract_row_payload(row)
            text = extract_row_text(row)
            raw_id = first_string(row, ID_FIELD_HINTS)
            original_id = safe_original_id(raw_id, f"row_{row_index:08d}")
            candidate = materialize_candidate(
                label=str(source["label"]),
                source=source,
                original_id=original_id,
                payload=payload,
                extension=extension,
                text=text,
            )
            if candidate:
                yield candidate


def request_huggingface_rows_page(
    params: Mapping[str, Any], headers: Mapping[str, str], timeout: int, attempts: int
) -> Mapping[str, Any]:
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                HF_ROWS_API_URL,
                params=dict(params),
                headers=dict(headers),
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise DatasetExpansionError("Hugging Face Rows API returned invalid JSON.")
            return payload
        except requests.RequestException as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            if not retryable or attempt >= attempts:
                raise
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"WARNING: Hugging Face Rows API request failed "
                f"({attempt}/{attempts}, status={status or 'network'}); retrying in {delay}s."
            )
            time.sleep(delay)
    raise DatasetExpansionError("Hugging Face Rows API retry loop ended unexpectedly.")


def iter_huggingface_repo_files(
    source: Mapping[str, Any], scan_limit: int, seed: int, timeout: int
) -> Iterator[CandidateAsset]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("Missing dependency 'huggingface_hub'.") from error

    repo_id = str(source["dataset"])
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    suffixes = tuple(str(value).lower() for value in source.get("patterns", [".pdf"]))
    matching = [name for name in files if name.lower().endswith(suffixes)]
    random.Random(seed).shuffle(matching)
    for path in matching[:scan_limit]:
        encoded_path = urllib.parse.quote(path, safe="/")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{encoded_path}?download=true"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.content
        extension = detect_extension(payload, Path(path).suffix)
        original_id = safe_original_id(path, f"file_{hash(path)}")
        candidate = materialize_candidate(
            label=str(source["label"]),
            source=source,
            original_id=original_id,
            payload=payload,
            extension=extension,
            text="",
        )
        if candidate:
            yield candidate


def iter_arxiv(source: Mapping[str, Any], scan_limit: int, seed: int, timeout: int) -> Iterator[CandidateAsset]:
    categories = list(source.get("categories") or ["cs.CV"])
    rng = random.Random(seed)
    rng.shuffle(categories)
    query = " OR ".join(f"cat:{category}" for category in categories)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    page_size = min(100, max(25, scan_limit))
    yielded = 0
    start = seed % 5000

    while yielded < scan_limit:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": query,
                "start": start,
                "max_results": min(page_size, scan_limit - yielded),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            timeout=timeout,
            headers={"User-Agent": "document-ai-classifier-dataset-expansion/1.0"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        entries = root.findall("atom:entry", namespace)
        if not entries:
            return
        for entry in entries:
            identifier = (entry.findtext("atom:id", default="", namespaces=namespace).rstrip("/").split("/")[-1])
            title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=namespace)).strip()
            summary = re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=namespace)).strip()
            pdf_url = ""
            for link in entry.findall("atom:link", namespace):
                if link.attrib.get("type") == "application/pdf" or link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            if not pdf_url or not identifier:
                continue
            pdf_response = requests.get(
                pdf_url,
                timeout=timeout,
                headers={"User-Agent": "document-ai-classifier-dataset-expansion/1.0"},
            )
            pdf_response.raise_for_status()
            candidate = materialize_candidate(
                label=str(source["label"]),
                source=source,
                original_id=safe_original_id(identifier, f"arxiv_{yielded}"),
                payload=pdf_response.content,
                extension=".pdf",
                text=f"{title}\n\nAbstract\n{summary}",
            )
            if candidate:
                yielded += 1
                yield candidate
                if yielded >= scan_limit:
                    return
        start += len(entries)
        time.sleep(3.0)


def iter_source_candidates(
    source: Mapping[str, Any], scan_limit: int, seed: int, timeout: int
) -> Iterator[CandidateAsset]:
    adapter = source["adapter"]
    if adapter == "huggingface_rows":
        yield from iter_huggingface_rows(source, scan_limit, seed)
    elif adapter == "huggingface_rows_api":
        yield from iter_huggingface_rows_api(source, scan_limit, seed, timeout)
    elif adapter == "huggingface_repo_files":
        yield from iter_huggingface_repo_files(source, scan_limit, seed, timeout)
    elif adapter == "arxiv_api":
        yield from iter_arxiv(source, scan_limit, seed, timeout)
    else:
        raise DatasetExpansionError(f"Unsupported source adapter: {adapter}")


def next_extra_number(label: str, metadata_rows: Iterable[Mapping[str, str]]) -> int:
    pattern = re.compile(rf"^{re.escape(label)}_extra_(\d+)$")
    numbers = []
    for row in metadata_rows:
        match = pattern.match(row.get("id", ""))
        if match:
            numbers.append(int(match.group(1)))
    for path in (RAW_DIR / label).glob(f"{label}_extra_*.*"):
        match = pattern.match(path.stem)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def create_backups() -> None:
    if not METADATA_BACKUP_PATH.exists():
        shutil.copy2(METADATA_PATH, METADATA_BACKUP_PATH)
        print(f"Metadata backup: {METADATA_BACKUP_PATH}")
    else:
        print(f"Metadata backup already exists, preserving it: {METADATA_BACKUP_PATH}")

    SPLITS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "validation.csv", "test.csv"):
        source = SPLITS_DIR / name
        destination = SPLITS_BACKUP_DIR / name
        if not source.exists():
            raise FileNotFoundError(f"Missing split before expansion: {source}")
        if not destination.exists():
            shutil.copy2(source, destination)
    print(f"Split backup: {SPLITS_BACKUP_DIR}")


def add_report_row(
    rows: list[dict[str, object]],
    *,
    candidate_path: str,
    label: str,
    source: str,
    decision: str,
    reason: str,
    similar_to: str = "",
    similarity_score: float | str = "",
) -> None:
    rows.append(
        {
            "candidate_path": candidate_path,
            "label": label,
            "source": source,
            "decision": decision,
            "reason": reason,
            "similar_to": similar_to,
            "similarity_score": (
                f"{similarity_score:.6f}" if isinstance(similarity_score, float) else similarity_score
            ),
        }
    )


def stage_remote_documents(
    *,
    plan: Mapping[str, Mapping[str, Any]],
    sources: list[dict[str, Any]],
    staging_root: Path,
    metadata_rows: list[dict[str, str]],
    fingerprint_index: Any,
    duplicate_rows: list[dict[str, object]],
    args: argparse.Namespace,
    existing_documents: Iterable[AcceptedDocument] = (),
) -> tuple[list[AcceptedDocument], Counter[str], list[str]]:
    source_by_name = {source["name"]: source for source in sources}
    existing_documents = list(existing_documents)
    next_numbers = {label: next_extra_number(label, metadata_rows) for label in CLASS_NAMES}
    for document in existing_documents:
        match = re.search(r"(\d+)$", document.document_id)
        if match:
            next_numbers[document.label] = max(next_numbers[document.label], int(match.group(1)) + 1)
    accepted: list[AcceptedDocument] = []
    skipped = Counter()
    warnings: list[str] = []
    known_originals_by_source: dict[str, set[str]] = defaultdict(set)
    for row in read_existing_source_rows():
        if row.get("is_augmented", "").casefold() not in {"true", "1", "yes"}:
            known_originals_by_source[row.get("source_name", "")].add(row.get("original_id", ""))
    for document in existing_documents:
        if not document.is_augmented and document.original_id:
            known_originals_by_source[document.source_name].add(document.original_id)
    staged_by_source = Counter(
        document.source_name for document in existing_documents if not document.is_augmented
    )

    for label in CLASS_NAMES:
        for source_name, quota in plan[label]["quotas"].items():
            if quota <= 0:
                continue
            already_staged = staged_by_source[source_name]
            remaining_quota = max(0, int(quota) - already_staged)
            if remaining_quota == 0:
                print(f"\n[{label}] {source_name}: already staged {already_staged}/{quota}, skipping download")
                continue
            source = source_by_name[source_name]
            accepted_for_source = 0
            scan_multiplier = max(1, int(source.get("scan_multiplier", 6)))
            already_seen = len(known_originals_by_source[source_name])
            scan_limit = max(
                (remaining_quota + already_seen) * scan_multiplier,
                remaining_quota + already_seen + 25,
            )
            print(
                f"\n[{label}] {source_name}: target remaining {remaining_quota} "
                f"(already staged {already_staged}/{quota}), scan limit {scan_limit}"
            )
            try:
                iterator = iter_source_candidates(
                    source,
                    scan_limit=scan_limit,
                    seed=args.seed + sum(ord(char) for char in source_name),
                    timeout=args.request_timeout,
                )
                for asset in tqdm(iterator, total=remaining_quota, desc=source_name, unit="accepted"):
                    if accepted_for_source >= remaining_quota:
                        break
                    if asset.original_id in known_originals_by_source[source_name]:
                        skipped[label] += 1
                        add_report_row(
                            duplicate_rows,
                            candidate_path=f"{source_name}:{asset.original_id}",
                            label=label,
                            source=source_name,
                            decision="skipped",
                            reason="already_added_source_original_id",
                        )
                        continue
                    number = next_numbers[label]
                    document_id = f"{label}_extra_{number:04d}"
                    final_path = RAW_DIR / label / f"{document_id}{asset.extension}"
                    validate_training_path(PROJECT_ROOT, final_path)
                    staging_path = staging_root / "raw" / label / final_path.name
                    staging_path.parent.mkdir(parents=True, exist_ok=True)
                    staging_path.write_bytes(asset.payload)

                    try:
                        fingerprint = build_fingerprint_record(
                            key=document_id,
                            raw_path=staging_path,
                            label=label,
                            source=source_name,
                            text_hint=asset.text,
                            group_id=document_id,
                        )
                        fingerprint.path = relative_project_path(PROJECT_ROOT, final_path)
                        match = fingerprint_index.find_duplicate(fingerprint)
                    except Exception as error:
                        staging_path.unlink(missing_ok=True)
                        skipped[label] += 1
                        add_report_row(
                            duplicate_rows,
                            candidate_path=f"{source_name}:{asset.original_id}",
                            label=label,
                            source=source_name,
                            decision="skipped",
                            reason=f"fingerprint_error: {error}",
                        )
                        continue

                    if match:
                        staging_path.unlink(missing_ok=True)
                        skipped[label] += 1
                        add_report_row(
                            duplicate_rows,
                            candidate_path=f"{source_name}:{asset.original_id}",
                            label=label,
                            source=source_name,
                            decision="skipped",
                            reason=match.reason,
                            similar_to=match.similar_to,
                            similarity_score=match.similarity_score,
                        )
                        continue

                    fingerprint_index.add(fingerprint)
                    accepted.append(
                        AcceptedDocument(
                            document_id=document_id,
                            label=label,
                            source_name=source_name,
                            source_locator=asset.source_locator,
                            original_id=asset.original_id,
                            raw_staging_path=staging_path,
                            raw_final_path=final_path,
                            text_hint=asset.text,
                        )
                    )
                    accepted_for_source += 1
                    known_originals_by_source[source_name].add(asset.original_id)
                    next_numbers[label] += 1
                    add_report_row(
                        duplicate_rows,
                        candidate_path=relative_project_path(PROJECT_ROOT, final_path),
                        label=label,
                        source=source_name,
                        decision="accepted",
                        reason="unique_exact_visual_and_text_fingerprints",
                    )
                    if accepted_for_source % 25 == 0:
                        write_staging_manifest(existing_documents + accepted, staging_root)
            except Exception as error:
                warning = (
                    f"Source {source_name} failed after {accepted_for_source}/{remaining_quota} "
                    f"new documents: {error}"
                )
                warnings.append(warning)
                print(f"WARNING: {warning}")
            finally:
                write_staging_manifest(existing_documents + accepted, staging_root)
            if accepted_for_source < remaining_quota:
                supplied = already_staged + accepted_for_source
                warnings.append(f"Source {source_name} supplied {supplied}/{quota} accepted documents.")
    return accepted, skipped, warnings


def augment_image(image: Image.Image, augmentation_type: str, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    if augmentation_type == "brightness_low":
        return ImageEnhance.Brightness(image).enhance(rng.uniform(0.78, 0.90))
    if augmentation_type == "slight_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.45, 0.85)))
    if augmentation_type == "low_contrast":
        return ImageEnhance.Contrast(image).enhance(rng.uniform(0.72, 0.88))
    if augmentation_type == "slight_rotation":
        angle = rng.choice((-1, 1)) * rng.uniform(1.0, 3.0)
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
    if augmentation_type == "screenshot_crop":
        width, height = image.size
        crop_x = max(1, int(width * rng.uniform(0.008, 0.025)))
        crop_y = max(1, int(height * rng.uniform(0.008, 0.025)))
        cropped = image.crop((crop_x, crop_y, width - crop_x, height - crop_y))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)
    if augmentation_type == "mild_noise":
        array = np.asarray(image, dtype=np.int16)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 4.0, array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")
    return image


def stage_augmentations(
    *,
    plan: Mapping[str, Mapping[str, Any]],
    accepted_real: list[AcceptedDocument],
    metadata_rows: list[dict[str, str]],
    staging_root: Path,
    fingerprint_index: Any,
    duplicate_rows: list[dict[str, object]],
    args: argparse.Namespace,
    existing_augmented: Iterable[AcceptedDocument] = (),
) -> tuple[list[AcceptedDocument], Counter[str]]:
    rng = random.Random(args.seed)
    existing_augmented = list(existing_augmented)
    types = (
        "brightness_low",
        "slight_blur",
        "low_contrast",
        "slight_rotation",
        "jpeg_compression",
        "screenshot_crop",
        "mild_noise",
    )
    next_numbers = {label: next_extra_number(label, metadata_rows) for label in CLASS_NAMES}
    for document in accepted_real + existing_augmented:
        match = re.search(r"(\d+)$", document.document_id)
        if match:
            next_numbers[document.label] = max(next_numbers[document.label], int(match.group(1)) + 1)

    augmented: list[AcceptedDocument] = []
    skipped = Counter()
    for label in CLASS_NAMES:
        label_originals = [document for document in accepted_real if document.label == label]
        existing_for_label = [
            document for document in existing_augmented if document.label == label
        ]
        used_parent_ids = {document.parent_id for document in existing_for_label if document.parent_id}
        originals = [
            document
            for document in label_originals
            if document.document_id not in used_parent_ids
        ]
        rng.shuffle(originals)
        requested = max(0, int(plan[label]["augmentation_target"]) - len(existing_for_label))
        max_total_allowed = int(
            len(label_originals)
            * args.augmentation_fraction
            / max(1 - args.augmentation_fraction, 0.01)
        )
        max_remaining = max(0, max_total_allowed - len(existing_for_label))
        count = min(requested, max_remaining, len(originals))
        accepted_for_label = 0
        for index, original in enumerate(originals):
            if accepted_for_label >= count:
                break
            augmentation_type = types[index % len(types)]
            from src.dataset_expansion import document_preview

            preview = document_preview(original.raw_staging_path)
            if preview is None:
                skipped[label] += 1
                continue
            changed = augment_image(preview, augmentation_type, rng)
            preview.close()

            number = next_numbers[label]
            document_id = f"{label}_extra_{number:04d}"
            final_path = RAW_DIR / label / f"{document_id}.jpg"
            staging_path = staging_root / "raw" / label / final_path.name
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            quality = rng.randint(52, 78) if augmentation_type == "jpeg_compression" else 88
            changed.save(staging_path, format="JPEG", quality=quality, optimize=True)
            changed.close()

            fingerprint = build_fingerprint_record(
                key=document_id,
                raw_path=staging_path,
                label=label,
                source=original.source_name,
                text_hint=original.text_hint,
                group_id=original.document_id,
            )
            fingerprint.path = relative_project_path(PROJECT_ROOT, final_path)
            match = fingerprint_index.find_duplicate(
                fingerprint,
                ignore={original.document_id, relative_project_path(PROJECT_ROOT, original.raw_final_path)},
            )
            if match:
                staging_path.unlink(missing_ok=True)
                skipped[label] += 1
                add_report_row(
                    duplicate_rows,
                    candidate_path=relative_project_path(PROJECT_ROOT, final_path),
                    label=label,
                    source=original.source_name,
                    decision="skipped",
                    reason=match.reason,
                    similar_to=match.similar_to,
                    similarity_score=match.similarity_score,
                )
                continue

            fingerprint_index.add(fingerprint)
            augmented.append(
                AcceptedDocument(
                    document_id=document_id,
                    label=label,
                    source_name=original.source_name,
                    source_locator=original.source_locator,
                    original_id=original.document_id,
                    raw_staging_path=staging_path,
                    raw_final_path=final_path,
                    text_hint=original.text_hint,
                    is_augmented=True,
                    augmentation_type=augmentation_type,
                    parent_id=original.document_id,
                )
            )
            next_numbers[label] += 1
            accepted_for_label += 1
            add_report_row(
                duplicate_rows,
                candidate_path=relative_project_path(PROJECT_ROOT, final_path),
                label=label,
                source=original.source_name,
                decision="accepted",
                reason="controlled_augmentation_grouped_with_original",
                similar_to=original.document_id,
            )
            if len(augmented) % 25 == 0:
                write_staging_manifest(accepted_real + existing_augmented + augmented, staging_root)
        if accepted_for_label < count:
            print(
                f"WARNING: {label} augmentations supplied "
                f"{len(existing_for_label) + accepted_for_label}/"
                f"{plan[label]['augmentation_target']} accepted documents."
            )
        write_staging_manifest(accepted_real + existing_augmented + augmented, staging_root)
    return augmented, skipped


def staging_manifest_path(staging_root: Path) -> Path:
    return staging_root / STAGING_MANIFEST_NAME


def write_staging_manifest(documents: list[AcceptedDocument], staging_root: Path) -> None:
    hints_dir = staging_root / "text_hints"
    rows: list[dict[str, str]] = []
    for document in documents:
        hint_file = ""
        if len(normalize_text(document.text_hint)) >= 20:
            hints_dir.mkdir(parents=True, exist_ok=True)
            hint_path = hints_dir / f"{document.document_id}.txt"
            if not hint_path.exists():
                hint_path.write_text(document.text_hint, encoding="utf-8", errors="ignore")
            hint_file = str(hint_path.relative_to(staging_root))
        rows.append(
            {
                "id": document.document_id,
                "label": document.label,
                "source_name": document.source_name,
                "source_locator": document.source_locator,
                "original_id": document.original_id,
                "raw_file": str(document.raw_staging_path.relative_to(staging_root)),
                "raw_final_path": relative_project_path(PROJECT_ROOT, document.raw_final_path),
                "text_hint_file": hint_file,
                "is_augmented": str(document.is_augmented),
                "augmentation_type": document.augmentation_type,
                "parent_id": document.parent_id,
            }
        )
    atomic_write_csv(staging_manifest_path(staging_root), rows, STAGING_MANIFEST_FIELDS)


def load_staging_manifest(staging_root: Path) -> list[AcceptedDocument]:
    path = staging_manifest_path(staging_root)
    if not path.exists():
        return []
    rows = read_csv_rows(path, STAGING_MANIFEST_FIELDS)
    documents: list[AcceptedDocument] = []
    for row in rows:
        raw_staging_path = staging_root / row["raw_file"]
        if not raw_staging_path.exists():
            continue
        hint_path = staging_root / row["text_hint_file"] if row.get("text_hint_file") else None
        text_hint = (
            hint_path.read_text(encoding="utf-8", errors="ignore")
            if hint_path and hint_path.exists()
            else ""
        )
        documents.append(
            AcceptedDocument(
                document_id=row["id"],
                label=row["label"],
                source_name=row["source_name"],
                source_locator=row["source_locator"],
                original_id=row["original_id"],
                raw_staging_path=raw_staging_path,
                raw_final_path=project_path(PROJECT_ROOT, row["raw_final_path"]),
                text_hint=text_hint,
                is_augmented=row["is_augmented"].casefold() in {"true", "1", "yes"},
                augmentation_type=row["augmentation_type"],
                parent_id=row["parent_id"],
            )
        )
    return documents


def find_resume_staging() -> Path | None:
    candidates = [
        path
        for path in DATA_DIR.glob(".dataset_expansion-*")
        if path.is_dir() and (path / "raw").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    selected = candidates[0]
    if len(candidates) > 1:
        print(f"WARNING: found {len(candidates)} staging directories; using newest: {selected}")
    return selected


def extra_document_number(path: Path) -> int:
    match = re.search(r"_extra_(\d+)$", path.stem)
    return int(match.group(1)) if match else 10**12


def reconstruct_staging_documents(
    staging_root: Path,
    plan: Mapping[str, Mapping[str, Any]],
    sources: list[dict[str, Any]],
    *,
    write_result: bool = True,
) -> list[AcceptedDocument]:
    source_by_name = {source["name"]: source for source in sources}
    documents: list[AcceptedDocument] = []
    print("Reconstructing staging entries from existing raw staging files.")

    for label in CLASS_NAMES:
        label_root = staging_root / "raw" / label
        raw_files = sorted(
            (
                path
                for path in label_root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=extra_document_number,
        ) if label_root.exists() else []
        real_count = min(int(plan[label]["real_target"]), len(raw_files))
        source_sequence: list[str] = []
        for source_name, quota in plan[label]["quotas"].items():
            source_sequence.extend([source_name] * int(quota))
        if len(source_sequence) < real_count:
            source_sequence.extend(["resumed_staging_unknown"] * (real_count - len(source_sequence)))

        for index, raw_path in enumerate(raw_files):
            document_id = raw_path.stem
            is_augmented = index >= real_count
            source_name = (
                "resumed_staging_augmentation" if is_augmented else source_sequence[index]
            )
            source = source_by_name.get(source_name, {})
            processed_text = staging_root / "processed" / "texts" / f"{document_id}.txt"
            text_hint = (
                processed_text.read_text(encoding="utf-8", errors="ignore")
                if processed_text.exists()
                else ""
            )
            documents.append(
                AcceptedDocument(
                    document_id=document_id,
                    label=label,
                    source_name=source_name,
                    source_locator=str(source.get("url", "resumed existing staging")),
                    original_id=document_id if not is_augmented else "",
                    raw_staging_path=raw_path,
                    raw_final_path=RAW_DIR / label / raw_path.name,
                    text_hint=text_hint,
                    is_augmented=is_augmented,
                    augmentation_type="resumed_existing_augmentation" if is_augmented else "",
                    parent_id="",
                )
            )
    if write_result:
        write_staging_manifest(documents, staging_root)
    return documents


def load_resume_documents(
    staging_root: Path,
    plan: Mapping[str, Mapping[str, Any]],
    sources: list[dict[str, Any]],
) -> list[AcceptedDocument]:
    documents = load_staging_manifest(staging_root)
    raw_files = {
        path.resolve()
        for label in CLASS_NAMES
        for path in (staging_root / "raw" / label).rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    manifested_files = {document.raw_staging_path.resolve() for document in documents}
    if documents and raw_files != manifested_files:
        print(
            "Staging manifest is incomplete or stale; adding only untracked raw staging files."
        )
        reconstructed = reconstruct_staging_documents(
            staging_root, plan, sources, write_result=False
        )
        reconstructed_by_path = {
            document.raw_staging_path.resolve(): document for document in reconstructed
        }
        documents.extend(
            reconstructed_by_path[path]
            for path in sorted(raw_files - manifested_files, key=str)
            if path in reconstructed_by_path
        )
        write_staging_manifest(documents, staging_root)
    if not documents:
        documents = reconstruct_staging_documents(staging_root, plan, sources)
    counts = Counter(document.label for document in documents)
    print(f"Resuming staging: {staging_root}")
    print("Staged documents: " + ", ".join(f"{label}={counts[label]}" for label in CLASS_NAMES))
    return documents


def staging_satisfies_plan(
    documents: Iterable[AcceptedDocument], plan: Mapping[str, Mapping[str, Any]]
) -> bool:
    documents = list(documents)
    real_by_source = Counter(
        document.source_name for document in documents if not document.is_augmented
    )
    augmented_by_label = Counter(
        document.label for document in documents if document.is_augmented
    )
    for label in CLASS_NAMES:
        for source_name, quota in plan[label]["quotas"].items():
            if real_by_source[source_name] < int(quota):
                return False
        if augmented_by_label[label] < int(plan[label]["augmentation_target"]):
            return False
    return True


def add_staged_fingerprints(
    documents: Iterable[AcceptedDocument], fingerprint_index: Any
) -> list[str]:
    warnings: list[str] = []
    for document in documents:
        try:
            fingerprint = build_fingerprint_record(
                key=document.document_id,
                raw_path=document.raw_staging_path,
                label=document.label,
                source=document.source_name,
                text_hint=document.text_hint,
                group_id=document.parent_id or document.document_id,
            )
            fingerprint.path = relative_project_path(PROJECT_ROOT, document.raw_final_path)
            fingerprint_index.add(fingerprint)
        except Exception as error:
            warnings.append(
                f"Could not rebuild staged fingerprint for {document.document_id}: {error}"
            )
    return warnings


def infer_missing_augmentation_parents(
    documents: list[AcceptedDocument], staging_root: Path, successful_ids: set[str]
) -> list[str]:
    warnings: list[str] = []
    for label in CLASS_NAMES:
        originals = [
            document
            for document in documents
            if document.label == label
            and not document.is_augmented
            and document.document_id in successful_ids
        ]
        augmentations = [
            document
            for document in documents
            if document.label == label
            and document.is_augmented
            and not document.parent_id
            and document.document_id in successful_ids
        ]
        hashes: list[tuple[AcceptedDocument, int]] = []
        for original in originals:
            image_path = staged_processed_paths(staging_root, original.document_id)[0]
            if not image_path.exists():
                image_path = final_processed_paths(original.document_id)[0]
            try:
                with Image.open(image_path) as image:
                    hashes.append((original, perceptual_hash(image)))
            except Exception:
                continue

        for augmentation in augmentations:
            image_path = staged_processed_paths(staging_root, augmentation.document_id)[0]
            if not image_path.exists():
                image_path = final_processed_paths(augmentation.document_id)[0]
            try:
                with Image.open(image_path) as image:
                    augmentation_hash = perceptual_hash(image)
            except Exception:
                warnings.append(f"Could not fingerprint augmentation {augmentation.document_id}")
                continue
            if not hashes:
                warnings.append(f"No parent candidates for augmentation {augmentation.document_id}")
                continue
            parent, score = max(
                ((candidate, bit_similarity(augmentation_hash, candidate_hash)) for candidate, candidate_hash in hashes),
                key=lambda item: item[1],
            )
            if score < 0.85:
                warnings.append(
                    f"No reliable parent for {augmentation.document_id}; best similarity {score:.3f}"
                )
                continue
            augmentation.parent_id = parent.document_id
            augmentation.original_id = parent.document_id
            augmentation.source_name = parent.source_name
            augmentation.source_locator = parent.source_locator
    write_staging_manifest(documents, staging_root)
    return warnings


def staged_processed_paths(staging_root: Path, document_id: str) -> tuple[Path, Path, Path]:
    return (
        staging_root / "processed" / "images" / f"{document_id}.png",
        staging_root / "processed" / "texts" / f"{document_id}.txt",
        staging_root / "processed" / "ocr" / f"{document_id}.json",
    )


def final_processed_paths(document_id: str) -> tuple[Path, Path, Path]:
    return (
        PROCESSED_DIR / "images" / f"{document_id}.png",
        PROCESSED_DIR / "texts" / f"{document_id}.txt",
        PROCESSED_DIR / "ocr" / f"{document_id}.json",
    )


def processed_outputs_valid(paths: tuple[Path, Path, Path]) -> tuple[bool, str]:
    image_path, text_path, ocr_path = paths
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return False, f"missing outputs: {', '.join(missing)}"
    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as error:
        return False, f"invalid processed image: {error}"
    try:
        payload = json.loads(ocr_path.read_text(encoding="utf-8"))
        words = payload.get("words", [])
        boxes = payload.get("boxes", [])
        if not isinstance(words, list) or not isinstance(boxes, list) or len(words) != len(boxes):
            return False, "invalid OCR words/boxes"
    except Exception as error:
        return False, f"invalid OCR JSON: {error}"
    try:
        text = normalize_text(text_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as error:
        return False, f"invalid processed text: {error}"
    if len(text) < 20 and not words:
        return False, "processed text and OCR are both empty or too short"
    return True, ""


def metadata_row_for_document(document: AcceptedDocument) -> dict[str, str]:
    image_final, text_final, ocr_final = final_processed_paths(document.document_id)
    return {
        "id": document.document_id,
        "label": document.label,
        "raw_path": relative_project_path(PROJECT_ROOT, document.raw_final_path),
        "image_path": relative_project_path(PROJECT_ROOT, image_final),
        "text_path": relative_project_path(PROJECT_ROOT, text_final),
        "ocr_path": relative_project_path(PROJECT_ROOT, ocr_final),
    }


def source_row_for_document(document: AcceptedDocument) -> dict[str, str]:
    return {
        "id": document.document_id,
        "label": document.label,
        "raw_path": relative_project_path(PROJECT_ROOT, document.raw_final_path),
        "source_name": document.source_name,
        "source_url_or_dataset": document.source_locator,
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "original_id": document.original_id,
        "language": "",
        "is_synthetic": "False",
        "is_augmented": str(document.is_augmented),
        "augmentation_type": document.augmentation_type,
        "duplicate_check_status": "passed",
    }


def read_failure_rows() -> list[dict[str, str]]:
    if not PREPROCESSING_FAILURE_PATH.exists():
        return []
    return read_csv_rows(PREPROCESSING_FAILURE_PATH, FAILURE_FIELDS)


def persist_failure_rows(failures_by_id: Mapping[str, Mapping[str, object]]) -> None:
    atomic_write_csv(
        PREPROCESSING_FAILURE_PATH,
        [failures_by_id[key] for key in sorted(failures_by_id)],
        FAILURE_FIELDS,
    )


def terminate_worker_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def execute_worker_command(
    command: list[str], timeout_seconds: float
) -> tuple[bool, int | None, str, str, float]:
    popen_kwargs: dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    started = time.monotonic()
    process = subprocess.Popen(command, **popen_kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return False, process.returncode, stdout, stderr, time.monotonic() - started
    except subprocess.TimeoutExpired:
        terminate_worker_tree(process)
        return True, process.returncode, "", "", time.monotonic() - started


def run_preprocess_worker(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    try:
        if hasattr(fitz.TOOLS, "mupdf_display_warnings"):
            fitz.TOOLS.mupdf_display_warnings(False)
        if hasattr(fitz.TOOLS, "mupdf_display_errors"):
            fitz.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass

    raw_path = Path(job["raw_path"])
    image_path = Path(job["image_path"])
    text_path = Path(job["text_path"])
    ocr_path = Path(job["ocr_path"])
    process_file_to_outputs(raw_path, job["label"], image_path, text_path, ocr_path)

    text_hint_path = Path(job["text_hint_path"]) if job.get("text_hint_path") else None
    if text_hint_path and text_hint_path.exists():
        text_hint = text_hint_path.read_text(encoding="utf-8", errors="ignore")
        if len(normalize_text(text_hint)) >= 20:
            text_path.write_text(text_hint.strip(), encoding="utf-8", errors="ignore")

    valid, error = processed_outputs_valid((image_path, text_path, ocr_path))
    if not valid:
        raise RuntimeError(error)


def preprocess_one_with_timeout(
    document: AcceptedDocument,
    staging_root: Path,
    timeout_seconds: int,
) -> tuple[bool, str, str, float]:
    image_stage, text_stage, ocr_stage = staged_processed_paths(staging_root, document.document_id)
    jobs_dir = staging_root / "preprocessing_jobs"
    hints_dir = staging_root / "text_hints"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    hints_dir.mkdir(parents=True, exist_ok=True)
    hint_path = hints_dir / f"{document.document_id}.txt"
    if len(normalize_text(document.text_hint)) >= 20 and not hint_path.exists():
        hint_path.write_text(document.text_hint, encoding="utf-8", errors="ignore")

    job_path = jobs_dir / f"{document.document_id}.json"
    atomic_write_text(
        job_path,
        json.dumps(
            {
                "raw_path": str(document.raw_staging_path),
                "label": document.label,
                "image_path": str(image_stage),
                "text_path": str(text_stage),
                "ocr_path": str(ocr_stage),
                "text_hint_path": str(hint_path) if hint_path.exists() else "",
            },
            ensure_ascii=False,
        ),
    )

    command = [sys.executable, str(Path(__file__).resolve()), "--worker-job", str(job_path)]
    timed_out, return_code, stdout, stderr, elapsed = execute_worker_command(
        command, timeout_seconds
    )
    if timed_out:
        return False, "timeout", f"Exceeded {timeout_seconds} seconds", elapsed
    if return_code != 0:
        details = (stderr or stdout or f"worker exited with code {return_code}").strip()
        return False, "preprocessing_error", details[-4000:], elapsed
    valid, validation_error = processed_outputs_valid((image_stage, text_stage, ocr_stage))
    if not valid:
        return False, "invalid_processed_output", validation_error, elapsed
    return True, "", "", elapsed


def preprocess_staged_documents(
    documents: list[AcceptedDocument],
    staging_root: Path,
    duplicate_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[AcceptedDocument],
    Counter[str],
    list[dict[str, str]],
]:
    metadata_rows: list[dict[str, str]] = []
    source_rows: list[dict[str, str]] = []
    successful: list[AcceptedDocument] = []
    failed = Counter()
    failures_by_id = {row["id"]: dict(row) for row in read_failure_rows()}
    progress = tqdm(documents, desc="Preprocessing accepted documents", unit="document")

    for index, document in enumerate(progress, start=1):
        if index == 1 or index % 5 == 0:
            progress.set_description(f"Preprocessing {document.document_id}")
        if index == 1 or index % 25 == 0:
            tqdm.write(f"Processing {index}/{len(documents)}: {document.document_id}")

        stage_paths = staged_processed_paths(staging_root, document.document_id)
        final_paths = final_processed_paths(document.document_id)
        stage_valid, _ = processed_outputs_valid(stage_paths)
        final_valid, _ = processed_outputs_valid(final_paths)
        if args.skip_existing and (stage_valid or final_valid):
            metadata_rows.append(metadata_row_for_document(document))
            source_rows.append(source_row_for_document(document))
            successful.append(document)
            continue

        if document.document_id in failures_by_id and not args.retry_failed:
            failed[document.label] += 1
            tqdm.write(f"SKIPPED previous failure: {document.raw_staging_path}")
            continue

        success, reason, error_message, elapsed = preprocess_one_with_timeout(
            document,
            staging_root,
            args.per_document_timeout,
        )
        if not success:
            failed[document.label] += 1
            failure_row = {
                "id": document.document_id,
                "label": document.label,
                "raw_path": relative_project_path(PROJECT_ROOT, document.raw_final_path),
                "source": document.source_name,
                "reason": reason,
                "error_message": error_message,
                "elapsed_seconds": f"{elapsed:.3f}",
            }
            failures_by_id[document.document_id] = failure_row
            persist_failure_rows(failures_by_id)
            if reason == "timeout":
                tqdm.write(f"SKIPPED timeout: {document.raw_staging_path}")
            else:
                tqdm.write(f"SKIPPED {reason}: {document.raw_staging_path}")
            add_report_row(
                duplicate_rows,
                candidate_path=relative_project_path(PROJECT_ROOT, document.raw_final_path),
                label=document.label,
                source=document.source_name,
                decision="skipped",
                reason=reason,
            )
            continue

        if document.document_id in failures_by_id:
            failures_by_id.pop(document.document_id)
            persist_failure_rows(failures_by_id)
        metadata_rows.append(metadata_row_for_document(document))
        source_rows.append(source_row_for_document(document))
        successful.append(document)

    persist_failure_rows(failures_by_id)
    return metadata_rows, source_rows, successful, failed, list(failures_by_id.values())


def commit_files(
    documents: list[AcceptedDocument],
    staging_root: Path,
    *,
    preserve_staging: bool,
    skip_existing: bool,
) -> None:
    for document in documents:
        destinations = (
            (document.raw_staging_path, document.raw_final_path),
            (
                staging_root / "processed" / "images" / f"{document.document_id}.png",
                PROCESSED_DIR / "images" / f"{document.document_id}.png",
            ),
            (
                staging_root / "processed" / "texts" / f"{document.document_id}.txt",
                PROCESSED_DIR / "texts" / f"{document.document_id}.txt",
            ),
            (
                staging_root / "processed" / "ocr" / f"{document.document_id}.json",
                PROCESSED_DIR / "ocr" / f"{document.document_id}.json",
            ),
        )
        for source, destination in destinations:
            validate_training_path(PROJECT_ROOT, destination)
            if destination.exists():
                if not skip_existing:
                    raise FileExistsError(f"Refusing to overwrite existing expansion output: {destination}")
                if source.exists() and sha256_file(source) != sha256_file(destination):
                    raise FileExistsError(
                        f"Existing destination differs from staged file: {destination}"
                    )
                continue
            if not source.exists():
                raise FileNotFoundError(f"Missing staged file: {source}")

    for document in documents:
        moves = (
            (document.raw_staging_path, document.raw_final_path),
            (
                staging_root / "processed" / "images" / f"{document.document_id}.png",
                PROCESSED_DIR / "images" / f"{document.document_id}.png",
            ),
            (
                staging_root / "processed" / "texts" / f"{document.document_id}.txt",
                PROCESSED_DIR / "texts" / f"{document.document_id}.txt",
            ),
            (
                staging_root / "processed" / "ocr" / f"{document.document_id}.json",
                PROCESSED_DIR / "ocr" / f"{document.document_id}.json",
            ),
        )
        for source, destination in moves:
            if destination.exists() and skip_existing:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if preserve_staging:
                shutil.copy2(source, destination)
            else:
                os.replace(source, destination)


def read_existing_source_rows() -> list[dict[str, str]]:
    if not SOURCE_TRACKING_PATH.exists():
        return []
    with SOURCE_TRACKING_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {field: str(row.get(field, "") or "") for field in SOURCE_FIELDS}
            for row in csv.DictReader(handle)
        ]


def augmentation_parent_map(source_rows: Iterable[Mapping[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in source_rows:
        if str(row.get("is_augmented", "")).casefold() in {"true", "1", "yes"}:
            result[str(row["id"])] = str(row["original_id"])
    return result


def write_summary(
    *,
    old_counts: Mapping[str, int],
    new_metadata_rows: list[dict[str, str]],
    skipped: Mapping[str, int],
    duplicate_rows: list[dict[str, object]],
    splits: Mapping[str, list[dict[str, str]]],
    source_rows: list[dict[str, str]],
    preprocessing_failures: list[dict[str, str]],
    warnings: list[str],
    target: int,
) -> None:
    added_counts = class_counts(new_metadata_rows)
    duplicate_reasons = {
        "identical_sha256",
        "identical_normalized_text",
        "near_identical_first_page",
        "near_duplicate_text",
    }
    duplicate_counts = Counter(
        str(row["label"])
        for row in duplicate_rows
        if row["decision"] == "skipped" and str(row["reason"]) in duplicate_reasons
    )
    split_counts = {name: class_counts(rows) for name, rows in splits.items()}
    preprocessing_failure_counts = Counter(row.get("label", "") for row in preprocessing_failures)
    source_names = defaultdict(set)
    for row in source_rows:
        source_names[row["label"]].add(row["source_name"])

    summary_rows: list[dict[str, object]] = []
    lines = ["DATASET EXPANSION SUMMARY", "=" * 80]
    for label in CLASS_NAMES:
        new_count = old_counts[label] + added_counts[label]
        warning = "" if new_count >= target else f"Below target {target} by {target - new_count}"
        summary_rows.append(
            {
                "label": label,
                "old_count": old_counts[label],
                "added": added_counts[label],
                "skipped": skipped.get(label, 0),
                "duplicate_or_similar": duplicate_counts[label],
                "preprocessing_failed": preprocessing_failure_counts[label],
                "new_count": new_count,
                "train": split_counts["train"][label],
                "validation": split_counts["validation"][label],
                "test": split_counts["test"][label],
                "sources": "; ".join(sorted(source_names[label])),
                "warning": warning,
            }
        )
        lines.extend(
            [
                f"\n{label}",
                f"  old: {old_counts[label]}",
                f"  added: {added_counts[label]}",
                f"  skipped: {skipped.get(label, 0)}",
                f"  duplicate/similar: {duplicate_counts[label]}",
                f"  preprocessing failed: {preprocessing_failure_counts[label]}",
                f"  new: {new_count}",
                f"  split train/validation/test: {split_counts['train'][label]}/"
                f"{split_counts['validation'][label]}/{split_counts['test'][label]}",
                f"  sources: {', '.join(sorted(source_names[label])) or 'none'}",
                f"  warning: {warning or 'none'}",
            ]
        )
    if warnings:
        lines.extend(["\nWARNINGS", *[f"- {warning}" for warning in warnings]])
    lines.extend(
        [
            "\nPREPROCESSING FAILURES",
            f"Total: {sum(preprocessing_failure_counts.values())}",
            f"Report: {PREPROCESSING_FAILURE_PATH}",
        ]
    )

    atomic_write_csv(
        RESULTS_DIR / "expansion_summary.csv",
        summary_rows,
        (
            "label",
            "old_count",
            "added",
            "skipped",
            "duplicate_or_similar",
            "preprocessing_failed",
            "new_count",
            "train",
            "validation",
            "test",
            "sources",
            "warning",
        ),
    )
    atomic_write_text(RESULTS_DIR / "expansion_summary.txt", "\n".join(lines) + "\n")


def real_run(
    args: argparse.Namespace,
    metadata_rows: list[dict[str, str]],
    sources: list[dict[str, Any]],
    plan: Mapping[str, Mapping[str, Any]],
) -> None:
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "Tesseract OCR is required before a real expansion run. "
            f"Configured command: {TESSERACT_COMMAND}"
        )
    create_backups()
    resume_staging = find_resume_staging() if args.resume else None
    resuming_existing = resume_staging is not None
    staging_root = resume_staging or Path(
        tempfile.mkdtemp(prefix=".dataset_expansion-", dir=DATA_DIR)
    )
    if args.resume and not resuming_existing:
        print(
            "WARNING: --resume was requested, but the previous staging directory no longer exists. "
            "Starting a new protected staging run; it will be retained."
        )
    duplicate_rows: list[dict[str, object]] = []
    if args.resume and (RESULTS_DIR / "duplicate_report.csv").exists():
        duplicate_rows.extend(
            read_csv_rows(RESULTS_DIR / "duplicate_report.csv", DUPLICATE_REPORT_FIELDS)
        )
    warnings: list[str] = []
    skipped = Counter()
    succeeded = False

    try:
        if resuming_existing:
            staged = load_resume_documents(staging_root, plan, sources)
            existing_ids = {row["id"] for row in metadata_rows}
            already_committed = [document for document in staged if document.document_id in existing_ids]
            if already_committed:
                print(f"Already present in metadata and skipped: {len(already_committed)}")
            staged = [document for document in staged if document.document_id not in existing_ids]
            for document in staged:
                add_report_row(
                    duplicate_rows,
                    candidate_path=relative_project_path(PROJECT_ROOT, document.raw_final_path),
                    label=document.label,
                    source=document.source_name,
                    decision="accepted",
                    reason="resumed_existing_staging",
                )
            if not staging_satisfies_plan(staged, plan):
                print("Partial staging detected; downloading only the missing planned documents.")
                print("Building fingerprints for existing and already staged documents...")
                fingerprint_index, fingerprint_failures = load_existing_fingerprint_index(
                    PROJECT_ROOT,
                    metadata_rows,
                    image_threshold=args.image_similarity_threshold,
                    text_threshold=args.text_similarity_threshold,
                )
                if fingerprint_failures:
                    first = fingerprint_failures[0]
                    raise DatasetExpansionError(
                        f"Could not fingerprint {len(fingerprint_failures)} existing documents. First: {first}"
                    )
                warnings.extend(add_staged_fingerprints(staged, fingerprint_index))

                existing_real = [document for document in staged if not document.is_augmented]
                existing_augmented = [document for document in staged if document.is_augmented]
                new_real, source_skips, source_warnings = stage_remote_documents(
                    plan=plan,
                    sources=sources,
                    staging_root=staging_root,
                    metadata_rows=metadata_rows,
                    fingerprint_index=fingerprint_index,
                    duplicate_rows=duplicate_rows,
                    args=args,
                    existing_documents=staged,
                )
                skipped.update(source_skips)
                warnings.extend(source_warnings)
                accepted_real = existing_real + new_real

                new_augmented, augmentation_skips = stage_augmentations(
                    plan=plan,
                    accepted_real=accepted_real,
                    metadata_rows=metadata_rows,
                    staging_root=staging_root,
                    fingerprint_index=fingerprint_index,
                    duplicate_rows=duplicate_rows,
                    args=args,
                    existing_augmented=existing_augmented,
                )
                skipped.update(augmentation_skips)
                staged = accepted_real + existing_augmented + new_augmented
                write_staging_manifest(staged, staging_root)
            else:
                print("Staging already satisfies the expansion plan; no downloads are needed.")
        else:
            print("\nBuilding fingerprints for the existing dataset...")
            fingerprint_index, fingerprint_failures = load_existing_fingerprint_index(
                PROJECT_ROOT,
                metadata_rows,
                image_threshold=args.image_similarity_threshold,
                text_threshold=args.text_similarity_threshold,
            )
            if fingerprint_failures:
                first = fingerprint_failures[0]
                raise DatasetExpansionError(
                    f"Could not fingerprint {len(fingerprint_failures)} existing documents. First: {first}"
                )

            accepted_real, source_skips, source_warnings = stage_remote_documents(
                plan=plan,
                sources=sources,
                staging_root=staging_root,
                metadata_rows=metadata_rows,
                fingerprint_index=fingerprint_index,
                duplicate_rows=duplicate_rows,
                args=args,
            )
            skipped.update(source_skips)
            warnings.extend(source_warnings)

            augmented, augmentation_skips = stage_augmentations(
                plan=plan,
                accepted_real=accepted_real,
                metadata_rows=metadata_rows,
                staging_root=staging_root,
                fingerprint_index=fingerprint_index,
                duplicate_rows=duplicate_rows,
                args=args,
            )
            skipped.update(augmentation_skips)
            staged = accepted_real + augmented
            write_staging_manifest(staged, staging_root)

        print(
            f"\nPreprocessing with a {args.per_document_timeout}s per-document timeout; "
            f"skip_existing={args.skip_existing}, retry_failed={args.retry_failed}"
        )
        (
            new_rows,
            new_source_rows,
            successful,
            preprocessing_failure_counts,
            preprocessing_failure_rows,
        ) = preprocess_staged_documents(
            staged,
            staging_root,
            duplicate_rows,
            args,
        )
        skipped.update(preprocessing_failure_counts)
        successful_ids = {document.document_id for document in successful}
        warnings.extend(
            infer_missing_augmentation_parents(staged, staging_root, successful_ids)
        )
        new_source_rows = [source_row_for_document(document) for document in successful]

        orphaned_augmentations = [
            document
            for document in successful
            if document.is_augmented and document.parent_id not in successful_ids
        ]
        for document in orphaned_augmentations:
            skipped[document.label] += 1
            add_report_row(
                duplicate_rows,
                candidate_path=relative_project_path(PROJECT_ROOT, document.raw_final_path),
                label=document.label,
                source=document.source_name,
                decision="skipped",
                reason="augmentation_parent_failed_preprocessing",
                similar_to=document.parent_id,
            )
        orphaned_ids = {document.document_id for document in orphaned_augmentations}
        successful = [document for document in successful if document.document_id not in orphaned_ids]
        new_rows = [row for row in new_rows if row["id"] not in orphaned_ids]
        new_source_rows = [row for row in new_source_rows if row["id"] not in orphaned_ids]

        over_limit_ids: set[str] = set()
        for label in CLASS_NAMES:
            real_successes = [
                document for document in successful if document.label == label and not document.is_augmented
            ]
            augmentation_successes = [
                document for document in successful if document.label == label and document.is_augmented
            ]
            max_augmentations = int(
                len(real_successes)
                * args.augmentation_fraction
                / max(1 - args.augmentation_fraction, 0.01)
            )
            for document in augmentation_successes[max_augmentations:]:
                over_limit_ids.add(document.document_id)
                skipped[label] += 1
                add_report_row(
                    duplicate_rows,
                    candidate_path=relative_project_path(PROJECT_ROOT, document.raw_final_path),
                    label=label,
                    source=document.source_name,
                    decision="skipped",
                    reason="augmentation_fraction_limit_after_preprocessing",
                    similar_to=document.parent_id,
                )
        successful = [document for document in successful if document.document_id not in over_limit_ids]
        new_rows = [row for row in new_rows if row["id"] not in over_limit_ids]
        new_source_rows = [row for row in new_source_rows if row["id"] not in over_limit_ids]
        successful_ids = {document.document_id for document in successful}
        new_source_rows = [row for row in new_source_rows if row["id"] in successful_ids]
        staged_ids = {document.document_id for document in staged}
        preprocessing_failure_rows = [
            row for row in preprocessing_failure_rows if row.get("id") in staged_ids
        ]

        existing_source_rows = read_existing_source_rows()
        all_source_rows = existing_source_rows + new_source_rows
        all_metadata_rows = metadata_rows + new_rows
        parents = augmentation_parent_map(all_source_rows)
        splits = group_aware_stratified_split(all_metadata_rows, parents, seed=args.seed)

        commit_files(
            successful,
            staging_root,
            preserve_staging=args.resume or args.keep_staging,
            skip_existing=args.skip_existing,
        )
        atomic_write_csv(METADATA_PATH, all_metadata_rows, METADATA_FIELDS)
        atomic_write_csv(SOURCE_TRACKING_PATH, all_source_rows, SOURCE_FIELDS)
        for split_name, split_rows in splits.items():
            atomic_write_csv(SPLITS_DIR / f"{split_name}.csv", split_rows, METADATA_FIELDS)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(
            RESULTS_DIR / "duplicate_report.csv", duplicate_rows, DUPLICATE_REPORT_FIELDS
        )
        write_summary(
            old_counts=class_counts(metadata_rows),
            new_metadata_rows=new_rows,
            skipped=skipped,
            duplicate_rows=duplicate_rows,
            splits=splits,
            source_rows=new_source_rows,
            preprocessing_failures=preprocessing_failure_rows,
            warnings=warnings,
            target=args.target_per_class,
        )
        succeeded = True
        print("\nExpansion completed and committed.")
        print(f"Metadata: {METADATA_PATH}")
        print(f"Source tracking: {SOURCE_TRACKING_PATH}")
        print(f"Duplicate report: {RESULTS_DIR / 'duplicate_report.csv'}")
        print(f"Summary: {RESULTS_DIR / 'expansion_summary.txt'}")
        print(f"Preprocessing failures: {len(preprocessing_failure_rows)}")
        print(f"Failure report: {PREPROCESSING_FAILURE_PATH}")
    finally:
        remove_staging = succeeded and not args.keep_staging and not args.resume
        if remove_staging and staging_root.exists():
            expected_parent = DATA_DIR.resolve()
            if staging_root.resolve().parent != expected_parent or not staging_root.name.startswith(
                ".dataset_expansion-"
            ):
                raise DatasetExpansionError(f"Refusing to remove unexpected staging path: {staging_root}")
            shutil.rmtree(staging_root)
        elif staging_root.exists():
            print(f"Staging retained for inspection: {staging_root}")


def main() -> None:
    args = parse_args()
    if args.worker_job:
        run_preprocess_worker(args.worker_job)
        return
    metadata_rows = read_csv_rows(METADATA_PATH, METADATA_FIELDS)
    validate_metadata(metadata_rows)
    current = class_counts(metadata_rows)
    sources, catalog_note = load_source_catalog(args.source_config, set(args.source))
    plan = build_plan(
        current,
        sources,
        args.target_per_class,
        args.augmentation_fraction,
        args.max_source_share,
    )

    print(f"Project: {PROJECT_ROOT}")
    print(f"Source catalog: {args.source_config}")
    if catalog_note:
        print(f"SOURCE NOTICE: {catalog_note}")
    print_plan(plan, sources, args.dry_run)
    if args.dry_run:
        return
    if not args.resume and not any(item["additions"] for item in plan.values()):
        print("All classes already meet the requested target; nothing to add.")
        return
    real_run(args, metadata_rows, sources, plan)


if __name__ == "__main__":
    main()
