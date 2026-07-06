import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    import pytesseract

    TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if TESSERACT_EXE.exists():
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)

    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    TESSERACT_AVAILABLE = False

try:
    from docx import Document

    DOCX_AVAILABLE = True
except Exception:
    Document = None
    DOCX_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
TEXTS_DIR = PROCESSED_DIR / "texts"
OCR_DIR = PROCESSED_DIR / "ocr"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
FAILED_REPORT_PATH = PROJECT_ROOT / "results" / "preprocessing" / "failed_text_extraction.csv"

CLASSES = ["invoice", "cv", "contract", "email", "scientific"]
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".html", ".htm", ".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MIN_TEXT_CHARS = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw documents into images, text, and OCR JSON.")
    parser.add_argument(
        "--retry-empty-only",
        action="store_true",
        help="Only retry metadata rows whose text_path is empty or shorter than 20 characters.",
    )
    return parser.parse_args()


def ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = str(text or "").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_html(html: str) -> str:
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    return clean_text(html)


def resolve_project_path(value):
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_text_length(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return len(clean_text(path.read_text(encoding="utf-8", errors="ignore")))


def needs_text_retry(text_path: Path) -> bool:
    return read_text_length(text_path) < MIN_TEXT_CHARS


def empty_ocr_payload(label: str):
    return {
        "label": label,
        "words": [],
        "boxes": [],
        "confidences": [],
        "page_indices": [],
    }


def run_ocr_on_image(image: Image.Image, label: str, page_index: int = 0):
    payload = empty_ocr_payload(label)

    if not TESSERACT_AVAILABLE:
        return "", payload

    try:
        image = image.convert("RGB")
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return "", payload

    words_for_text = []
    for index, raw_word in enumerate(ocr_data.get("text", [])):
        word = clean_text(raw_word)
        if not word:
            continue

        x = int(ocr_data["left"][index])
        y = int(ocr_data["top"][index])
        w = int(ocr_data["width"][index])
        h = int(ocr_data["height"][index])
        confidence = ocr_data.get("conf", [""])[index]

        payload["words"].append(word)
        payload["boxes"].append([x, y, x + w, y + h])
        payload["confidences"].append(confidence)
        payload["page_indices"].append(page_index)
        words_for_text.append(word)

    return clean_text(" ".join(words_for_text)), payload


def merge_ocr_payloads(label: str, payloads):
    merged = empty_ocr_payload(label)
    for payload in payloads:
        merged["words"].extend(payload.get("words", []))
        merged["boxes"].extend(payload.get("boxes", []))
        merged["confidences"].extend(payload.get("confidences", []))
        merged["page_indices"].extend(payload.get("page_indices", []))
    return merged


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clean_text(text), encoding="utf-8", errors="ignore")


def write_ocr_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_image(image: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def render_pdf_page(page, zoom=2.0):
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)


def pdf_to_image_text_ocr(input_path: Path, image_out: Path, text_out: Path, ocr_out: Path, label: str):
    document = fitz.open(input_path)
    try:
        if document.page_count < 1:
            write_text(text_out, "")
            write_ocr_json(ocr_out, empty_ocr_payload(label))
            return

        first_page_image = render_pdf_page(document.load_page(0))
        save_image(first_page_image, image_out)

        embedded_text = clean_text("\n".join(page.get_text("text") for page in document))
        if len(embedded_text) >= MIN_TEXT_CHARS:
            write_text(text_out, embedded_text)
            _, payload = run_ocr_on_image(first_page_image, label, page_index=0)
            write_ocr_json(ocr_out, payload)
            return

        page_texts = []
        payloads = []
        for page_index in range(document.page_count):
            page_image = render_pdf_page(document.load_page(page_index))
            page_text, payload = run_ocr_on_image(page_image, label, page_index=page_index)
            if page_text:
                page_texts.append(page_text)
            payloads.append(payload)

        write_text(text_out, "\n".join(page_texts))
        write_ocr_json(ocr_out, merge_ocr_payloads(label, payloads))
    finally:
        document.close()


def image_to_image_text_ocr(input_path: Path, image_out: Path, text_out: Path, ocr_out: Path, label: str):
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        save_image(image, image_out)
        text, payload = run_ocr_on_image(image, label, page_index=0)

    write_text(text_out, text)
    write_ocr_json(ocr_out, payload)


def text_to_image_text_ocr(input_path: Path, image_out: Path, text_out: Path, ocr_out: Path, label: str, is_html=False):
    raw = input_path.read_text(encoding="utf-8", errors="ignore")
    text = strip_html(raw) if is_html else clean_text(raw)
    write_text(text_out, text)
    render_text_to_image(text, image_out)

    with Image.open(image_out) as image:
        _, payload = run_ocr_on_image(image.convert("RGB"), label, page_index=0)
    write_ocr_json(ocr_out, payload)


def docx_to_image_text_ocr(input_path: Path, image_out: Path, text_out: Path, ocr_out: Path, label: str):
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx nije instaliran. Instaliraj: pip install python-docx")

    doc = Document(input_path)
    text = clean_text("\n".join(paragraph.text for paragraph in doc.paragraphs))
    write_text(text_out, text)
    render_text_to_image(text, image_out)

    with Image.open(image_out) as image:
        _, payload = run_ocr_on_image(image.convert("RGB"), label, page_index=0)
    write_ocr_json(ocr_out, payload)


def render_text_to_image(text: str, image_out: Path):
    width, height = 1200, 1600
    margin = 60
    line_spacing = 8
    font_size = 28

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    words = clean_text(text).split()
    lines = []
    current_line = ""

    for word in words:
        test_line = (current_line + " " + word).strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= width - 2 * margin:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    y = margin
    for line in lines:
        if y > height - margin:
            break
        draw.text((margin, y), line, fill="black", font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += bbox[3] - bbox[1] + line_spacing

    save_image(image, image_out)


def process_file_to_outputs(input_path: Path, label: str, image_out: Path, text_out: Path, ocr_out: Path):
    ext = input_path.suffix.lower()

    if ext == ".pdf":
        pdf_to_image_text_ocr(input_path, image_out, text_out, ocr_out, label)
    elif ext in IMAGE_EXTENSIONS:
        image_to_image_text_ocr(input_path, image_out, text_out, ocr_out, label)
    elif ext == ".txt":
        text_to_image_text_ocr(input_path, image_out, text_out, ocr_out, label, is_html=False)
    elif ext in {".html", ".htm"}:
        text_to_image_text_ocr(input_path, image_out, text_out, ocr_out, label, is_html=True)
    elif ext == ".docx":
        docx_to_image_text_ocr(input_path, image_out, text_out, ocr_out, label)
    else:
        raise ValueError(f"Nepodrzan file type: {input_path}")


def process_file(input_path: Path, label: str, index: int):
    safe_id = f"{label}_{index:04d}"

    image_out = IMAGES_DIR / f"{safe_id}.png"
    text_out = TEXTS_DIR / f"{safe_id}.txt"
    ocr_out = OCR_DIR / f"{safe_id}.json"

    process_file_to_outputs(input_path, label, image_out, text_out, ocr_out)

    return {
        "id": safe_id,
        "label": label,
        "raw_path": str(input_path.relative_to(PROJECT_ROOT)),
        "image_path": str(image_out.relative_to(PROJECT_ROOT)),
        "text_path": str(text_out.relative_to(PROJECT_ROOT)),
        "ocr_path": str(ocr_out.relative_to(PROJECT_ROOT)),
    }


def collect_files_for_class(label: str):
    class_dir = RAW_DIR / label

    if not class_dir.exists():
        print(f"UPOZORENJE: Ne postoji folder: {class_dir}")
        return []

    files = []
    for path in class_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


def read_metadata_rows():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata ne postoji: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames or []


def retry_empty_only():
    rows, _ = read_metadata_rows()
    retry_rows = []

    for row in rows:
        text_path = resolve_project_path(row.get("text_path", ""))
        if needs_text_retry(text_path):
            retry_rows.append(row)

    print(f"Retry-empty-only: pronadeno {len(retry_rows)} dokumenata za ponovnu obradu.")

    processed = 0
    for row in tqdm(retry_rows, desc="Retry OCR/text extraction"):
        label = row["label"]
        raw_path = resolve_project_path(row["raw_path"])
        image_out = resolve_project_path(row["image_path"])
        text_out = resolve_project_path(row["text_path"])
        ocr_out = resolve_project_path(row["ocr_path"])

        if not raw_path.exists():
            print(f"UPOZORENJE: raw file ne postoji za {row.get('id')}: {raw_path}")
            continue

        try:
            process_file_to_outputs(raw_path, label, image_out, text_out, ocr_out)
            processed += 1
        except Exception as error:
            print(f"GRESKA kod {row.get('id')}: {raw_path}")
            print(error)

    failed_rows = summarize_text_lengths(rows)
    write_failed_report(failed_rows)
    print_summary(processed, rows)


def summarize_text_lengths(rows):
    failed_rows = []
    for row in rows:
        text_path = resolve_project_path(row.get("text_path", ""))
        length = read_text_length(text_path)
        if length < MIN_TEXT_CHARS:
            failed_rows.append(
                {
                    "id": row.get("id", ""),
                    "label": row.get("label", ""),
                    "raw_path": row.get("raw_path", ""),
                    "text_path": row.get("text_path", ""),
                    "text_length": length,
                }
            )
    return failed_rows


def write_failed_report(failed_rows):
    FAILED_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_REPORT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "label", "raw_path", "text_path", "text_length"],
        )
        writer.writeheader()
        writer.writerows(failed_rows)


def print_summary(processed_count, rows):
    lengths_by_label = defaultdict(list)
    empty_by_label = defaultdict(int)

    for row in rows:
        label = row.get("label", "")
        text_path = resolve_project_path(row.get("text_path", ""))
        length = read_text_length(text_path)
        lengths_by_label[label].append(length)
        if length < MIN_TEXT_CHARS:
            empty_by_label[label] += 1

    print()
    print(f"Ponovno obradeno dokumenata: {processed_count}")
    print("Jos uvijek prazni/kratki tekstovi po labeli:")
    for label in CLASSES:
        print(f"{label}: {empty_by_label[label]}")

    print("Prosjecna duljina teksta po labeli:")
    for label in CLASSES:
        lengths = lengths_by_label[label]
        average = sum(lengths) / len(lengths) if lengths else 0.0
        print(f"{label}: {average:.2f}")

    print(f"Failed text extraction report: {FAILED_REPORT_PATH}")


def full_preprocess():
    rows = []

    for label in CLASSES:
        files = collect_files_for_class(label)
        print(f"{label}: pronadeno {len(files)} fileova")

        for index, file_path in enumerate(tqdm(files, desc=f"Processing {label}"), start=1):
            try:
                row = process_file(file_path, label, index)
                rows.append(row)
            except Exception as error:
                print(f"GRESKA kod filea: {file_path}")
                print(error)

    with METADATA_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "label", "raw_path", "image_path", "text_path", "ocr_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    failed_rows = summarize_text_lengths(rows)
    write_failed_report(failed_rows)
    print_summary(len(rows), rows)
    print()
    print(f"Gotovo. Metadata spremljen u: {METADATA_PATH}")


def main():
    args = parse_args()
    ensure_dirs()

    print(f"Tesseract executable path: {TESSERACT_EXE}")
    print(f"Tesseract executable exists: {TESSERACT_EXE.exists()}")
    print(f"pytesseract available: {TESSERACT_AVAILABLE}")
    print()

    if args.retry_empty_only:
        retry_empty_only()
    else:
        full_preprocess()


if __name__ == "__main__":
    main()
