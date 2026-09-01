import json
import shutil
from pathlib import Path

import fitz
from docx import Document
from PIL import Image, ImageDraw, ImageFont

from .document_conversion import (
    DocumentConversionError,
    convert_docx_to_pdf,
)
from .multipage_preprocess import prepare_inference_file_artifacts
from .preprocess import MIN_TEXT_CHARS, OCRProcessingError, clean_text, render_pdf_page, run_ocr_on_image


SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".docx"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _add_error(errors, category, message):
    errors.append(f"{category}: {message}")


def _save_uploaded_file(uploaded_file, temp_dir):
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(uploaded_file, (str, Path)):
        source_path = Path(uploaded_file)
        name = source_path.name
    else:
        source_path = None
        name = Path(str(getattr(uploaded_file, "name", "uploaded_document"))).name

    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Nepodržan format '{suffix}'. Podržano: {supported}")

    destination = temp_dir / f"original{suffix}"
    if source_path is not None:
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Upload datoteka nije pronađena: {source_path}")
        if source_path != destination.resolve():
            shutil.copy2(source_path, destination)
        return destination

    if hasattr(uploaded_file, "getbuffer"):
        data = bytes(uploaded_file.getbuffer())
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        raise TypeError("Upload objekt ne sadrži getbuffer() ni read().")

    destination.write_bytes(data)
    return destination


def _write_text(text, output_path, errors, source_label):
    text = clean_text(text)
    if len(text) < MIN_TEXT_CHARS:
        _add_error(
            errors,
            "Tekstualni input nije pripremljen",
            f"{source_label} nema dovoljno čitljivog teksta ({len(text)} znakova).",
        )
        return None

    output_path = Path(output_path)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _write_ocr_payload(payload, output_path):
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _load_render_font(font_size=28):
    candidates = (
        "DejaVuSans.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_text_to_image_with_layout(text, image_path):
    width, height = 1200, 1600
    margin = 60
    font = _load_render_font()
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    words = clean_text(text).split()

    payload = {
        "label": "unknown",
        "words": [],
        "boxes": [],
        "confidences": [],
        "page_indices": [],
    }

    x = margin
    y = margin
    line_height = max(34, draw.textbbox((0, 0), "Ag", font=font)[3] + 10)
    space_width = max(8, int(draw.textlength(" ", font=font)))

    for word in words:
        word_width = max(1, int(draw.textlength(word, font=font)))
        if x + word_width > width - margin:
            x = margin
            y += line_height
        if y + line_height > height - margin:
            break

        draw.text((x, y), word, fill="black", font=font)
        box = draw.textbbox((x, y), word, font=font)
        payload["words"].append(word)
        payload["boxes"].append([int(value) for value in box])
        payload["confidences"].append("generated")
        payload["page_indices"].append(0)
        x += word_width + space_width

    image_path = Path(image_path)
    image.save(image_path)
    return image_path, payload


def _synthetic_layout_payload(text, image_path):
    with Image.open(image_path) as image:
        width, height = image.size

    words = clean_text(text).split()[:512]
    margin_x = max(10, int(width * 0.05))
    margin_y = max(10, int(height * 0.05))
    usable_width = max(100, width - 2 * margin_x)
    line_height = max(18, int(height * 0.025))
    character_width = max(7, int(width * 0.008))

    payload = {
        "label": "unknown",
        "words": [],
        "boxes": [],
        "confidences": [],
        "page_indices": [],
    }
    x = margin_x
    y = margin_y
    for word in words:
        word_width = min(usable_width, max(character_width, len(word) * character_width))
        if x + word_width > width - margin_x:
            x = margin_x
            y += line_height
        if y + line_height > height - margin_y:
            break
        payload["words"].append(word)
        payload["boxes"].append([x, y, x + word_width, y + line_height])
        payload["confidences"].append("generated")
        payload["page_indices"].append(0)
        x += word_width + character_width
    return payload


def _ocr_image(image_path):
    with Image.open(image_path) as image:
        return run_ocr_on_image(image.convert("RGB"), "unknown", page_index=0)


def _prepare_layout_payload(image_path, fallback_text, ocr_path, errors):
    try:
        ocr_text, payload = _ocr_image(image_path)
    except OCRProcessingError as error:
        ocr_text = ""
        payload = None
        _add_error(errors, "OCR upozorenje", str(error))

    if payload and payload.get("words"):
        return _write_ocr_payload(payload, ocr_path), clean_text(ocr_text)

    if len(clean_text(fallback_text)) >= MIN_TEXT_CHARS:
        payload = _synthetic_layout_payload(fallback_text, image_path)
        if payload.get("words"):
            _add_error(
                errors,
                "OCR upozorenje",
                "Korišteni su generirani bounding boxovi jer OCR nije pronašao riječi.",
            )
            return _write_ocr_payload(payload, ocr_path), clean_text(ocr_text)

    _add_error(
        errors,
        "OCR/layout input nije pripremljen",
        "Nije moguće pripremiti riječi i bounding boxove za LayoutLMv3.",
    )
    return None, clean_text(ocr_text)


def _extract_pdf_embedded_text(pdf_path):
    document = fitz.open(str(pdf_path))
    try:
        return clean_text("\n".join(page.get_text("text") for page in document))
    finally:
        document.close()


def _ocr_pdf_text(pdf_path, prepared_artifacts=None):
    prepared_by_page = {
        int(artifact.page_index): artifact for artifact in (prepared_artifacts or [])
    }
    texts = []
    document = fitz.open(str(pdf_path))
    try:
        for page_index in range(document.page_count):
            if page_index in prepared_by_page:
                text = clean_text(" ".join(prepared_by_page[page_index].words))
                if text:
                    texts.append(text)
                continue
            image = render_pdf_page(document.load_page(page_index))
            page_text, _ = run_ocr_on_image(image, "unknown", page_index=page_index)
            if page_text:
                texts.append(page_text)
    finally:
        document.close()
    return clean_text("\n".join(texts))


def _store_page_artifacts(result, total_pages, selected_indices, artifacts):
    result["total_pages"] = int(total_pages)
    result["selected_page_indices"] = [int(index) for index in selected_indices]
    page_artifacts = []
    layout_page_artifacts = []
    for artifact in artifacts:
        if not artifact.image_path.is_file():
            continue
        page_payload = {
            "page_index": int(artifact.page_index),
            "total_pages": int(artifact.total_pages),
            "image_path": artifact.image_path,
            "ocr_path": artifact.ocr_path,
            "words": list(artifact.words),
            "boxes": list(artifact.boxes),
            "extraction_method": artifact.extraction_method,
            "layout_status": artifact.layout_status,
            "failure_reason": artifact.failure_reason,
        }
        page_artifacts.append(page_payload)
        if artifact.is_layout_valid and artifact.ocr_path.is_file():
            layout_page_artifacts.append(page_payload)

    result["page_artifacts"] = page_artifacts
    result["layout_page_artifacts"] = layout_page_artifacts
    result["analyzed_page_indices"] = [page["page_index"] for page in page_artifacts]
    result["layout_page_indices"] = [page["page_index"] for page in layout_page_artifacts]
    if page_artifacts:
        result["image_path"] = page_artifacts[0]["image_path"]
    if layout_page_artifacts:
        result["ocr_path"] = layout_page_artifacts[0]["ocr_path"]


def _prepare_shared_pages(result, temp_dir, document_path=None):
    source_path = Path(document_path or result["original_path"])
    total_pages, selected_indices, artifacts = prepare_inference_file_artifacts(
        source_path,
        Path(temp_dir) / "multipage",
        source_image_path=result.get("image_path"),
        source_ocr_path=result.get("ocr_path"),
    )
    _store_page_artifacts(result, total_pages, selected_indices, artifacts)
    invalid_layout_pages = [artifact for artifact in artifacts if not artifact.is_layout_valid]
    if invalid_layout_pages:
        page_numbers = ", ".join(str(artifact.page_index + 1) for artifact in invalid_layout_pages)
        _add_error(
            result["errors"],
            "OCR/layout input nije pripremljen",
            f"Preskocene stranice bez valjanog OCR-a: {page_numbers}.",
        )
    return artifacts


def _prepare_pdf(result, temp_dir):
    errors = result["errors"]
    pdf_path = result["original_path"]
    result["pdf_path"] = pdf_path

    artifacts = []
    try:
        artifacts = _prepare_shared_pages(result, temp_dir, pdf_path)
    except Exception as error:
        _add_error(errors, "Vizualni/OCR input nije pripremljen", str(error))

    try:
        embedded_text = _extract_pdf_embedded_text(pdf_path)
    except Exception as error:
        embedded_text = ""
        _add_error(errors, "Tekstualni input nije pripremljen", f"PDF tekst nije čitljiv: {error}")

    final_text = embedded_text
    if len(final_text) < MIN_TEXT_CHARS:
        try:
            final_text = _ocr_pdf_text(pdf_path, prepared_artifacts=artifacts)
        except OCRProcessingError as error:
            _add_error(errors, "Tekstualni input nije pripremljen", str(error))

    result["text_path"] = _write_text(
        final_text,
        temp_dir / "document_text.txt",
        errors,
        "PDF",
    )

def _prepare_image(result, temp_dir):
    errors = result["errors"]
    result["image_path"] = result["original_path"]
    try:
        ocr_text, payload = _ocr_image(result["image_path"])
    except OCRProcessingError as error:
        _add_error(errors, "Tekstualni input nije pripremljen", str(error))
        _add_error(errors, "OCR/layout input nije pripremljen", str(error))
        return

    result["text_path"] = _write_text(
        ocr_text,
        temp_dir / "document_text.txt",
        errors,
        "Slika",
    )
    if payload.get("words"):
        result["ocr_path"] = _write_ocr_payload(payload, temp_dir / "document_ocr.json")
    else:
        _add_error(
            errors,
            "OCR/layout input nije pripremljen",
            "OCR nije pronašao riječi ni bounding boxove na slici.",
        )


def _prepare_docx(result, temp_dir):
    errors = result["errors"]
    try:
        document = Document(result["original_path"])
        text = clean_text("\n".join(paragraph.text for paragraph in document.paragraphs))
    except Exception as error:
        text = ""
        _add_error(errors, "Tekstualni input nije pripremljen", f"DOCX tekst nije čitljiv: {error}")

    result["text_path"] = _write_text(
        text,
        temp_dir / "document_text.txt",
        errors,
        "DOCX",
    )

    try:
        result["pdf_path"] = convert_docx_to_pdf(result["original_path"], temp_dir)
    except DocumentConversionError as error:
        _add_error(
            errors,
            "Vizualni input nije pripremljen",
            f"Nije moguće pretvoriti DOCX u sliku: {error}",
        )
        return


def _prepare_txt(result, temp_dir):
    errors = result["errors"]
    text = clean_text(result["original_path"].read_text(encoding="utf-8", errors="ignore"))
    result["text_path"] = _write_text(
        text,
        temp_dir / "document_text.txt",
        errors,
        "TXT",
    )
    if result["text_path"] is None:
        return

    image_path, payload = _render_text_to_image_with_layout(
        text,
        temp_dir / "document_first_page.png",
    )
    result["image_path"] = image_path
    if payload.get("words"):
        result["ocr_path"] = _write_ocr_payload(payload, temp_dir / "document_ocr.json")
    else:
        _add_error(
            errors,
            "OCR/layout input nije pripremljen",
            "TXT nije moguće rasporediti na vizualnu stranicu.",
        )


def prepare_document_for_models(uploaded_file, temp_dir) -> dict:
    temp_dir = Path(temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_path = _save_uploaded_file(uploaded_file, temp_dir)
    suffix = original_path.suffix.lower()

    result = {
        "original_path": original_path,
        "suffix": suffix,
        "pdf_path": None,
        "image_path": None,
        "text_path": None,
        "ocr_path": None,
        "total_pages": 0,
        "selected_page_indices": [],
        "analyzed_page_indices": [],
        "layout_page_indices": [],
        "page_artifacts": [],
        "layout_page_artifacts": [],
        "errors": [],
    }

    if suffix == ".pdf":
        _prepare_pdf(result, temp_dir)
    elif suffix in IMAGE_SUFFIXES:
        _prepare_image(result, temp_dir)
    elif suffix == ".docx":
        _prepare_docx(result, temp_dir)
    elif suffix == ".txt":
        _prepare_txt(result, temp_dir)

    if suffix != ".pdf" and (result.get("image_path") or result.get("pdf_path")):
        legacy_image_path = result.get("image_path")
        shared_source = result.get("pdf_path") or result["original_path"]
        try:
            _prepare_shared_pages(result, temp_dir, shared_source)
            if suffix in IMAGE_SUFFIXES:
                result["image_path"] = legacy_image_path
        except Exception as error:
            _add_error(
                result["errors"],
                "Multi-page input nije pripremljen",
                str(error),
            )

    return result
