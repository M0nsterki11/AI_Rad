"""Add privacy-safe Croatian documents to the existing training dataset.

The script writes one primary raw document per metadata row into the existing
class folders. Images, text and OCR are stored in the existing processed
folders. External evaluation folders are fingerprinted as exclusion data and
are never write targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import fitz
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm import tqdm


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_expansion import (  # noqa: E402
    CLASS_NAMES,
    METADATA_FIELDS,
    DatasetExpansionError,
    FingerprintIndex,
    atomic_write_csv,
    atomic_write_text,
    build_fingerprint_record,
    class_counts,
    group_aware_stratified_split,
    load_existing_fingerprint_index,
    normalize_text,
    project_path,
    read_csv_rows,
    relative_project_path,
    sha256_file,
    validate_training_path,
    visible_html_text,
)
from src.preprocess import (  # noqa: E402
    TESSERACT_AVAILABLE,
    process_file_to_outputs,
    run_ocr_on_image,
)


DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_PATH = DATA_DIR / "metadata.csv"
SPLITS_DIR = DATA_DIR / "splits"
SOURCE_TRACKING_PATH = DATA_DIR / "dataset_sources_extra.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "dataset_expansion"
HR_SUMMARY_PATH = RESULTS_DIR / "hr_dataset_summary.txt"
HR_SOURCES_PATH = RESULTS_DIR / "hr_dataset_sources.csv"
HR_DUPLICATES_PATH = RESULTS_DIR / "hr_duplicate_report.csv"

HRCAK_OAI_URL = "https://hrcak.srce.hr/oai/"
DATA_GOV_API_URL = "https://data.gov.hr/ckan/api/3/action/package_search"
USER_AGENT = "DocumentAIClassifier-DatasetResearch/1.0 (educational project)"
PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024

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
HR_DUPLICATE_FIELDS = (
    "candidate_id",
    "candidate_path",
    "label",
    "source_name",
    "decision",
    "reason",
    "similar_to",
    "similarity_score",
)
HR_SOURCE_NAMES = {
    "hrcak",
    "hrvatska_javna_nabava",
    "hr_synthetic_invoice",
    "hr_synthetic_cv",
    "hr_synthetic_gmail_like",
    "hr_synthetic_contract",
    "hr_synthetic_scientific",
}
ID_PREFIXES = {
    "invoice": "invoice_hr_synthetic",
    "cv": "cv_hr_synthetic",
    "contract": "contract_hr",
    "email": "email_hr_gmail_like",
    "scientific": "scientific_hr",
}
AUGMENTATION_TYPES = (
    "brightness_low",
    "slight_blur",
    "low_contrast",
    "jpeg_compression",
    "screenshot_crop",
    "slight_rotation",
)
SENSITIVE_FIELD_MARKERS = {
    "email",
    "e-mail",
    "telefon",
    "mobitel",
    "kontakt",
    "adresa",
    "ime i prezime",
    "ime osobe",
    "prezime",
    "osoba za kontakt",
    "potpis",
    "iban",
    "oib",
}


@dataclass(slots=True)
class RenderedDocument:
    image: Image.Image
    text: str
    words: list[str]
    boxes: list[list[int]]


@dataclass(slots=True)
class Candidate:
    label: str
    source_name: str
    source_locator: str
    original_id: str
    kind: str
    is_synthetic: bool
    data: dict[str, Any] = field(default_factory=dict)
    payload: bytes | None = None
    extension: str = ".pdf"


@dataclass(slots=True)
class PreparedDocument:
    document_id: str
    label: str
    source_name: str
    source_locator: str
    original_id: str
    language: str
    is_synthetic: bool
    is_augmented: bool
    augmentation_type: str
    parent_id: str
    raw_stage_path: Path
    image_stage_path: Path
    text_stage_path: Path
    ocr_stage_path: Path
    raw_final_path: Path


class RemoteAccessDenied(DatasetExpansionError):
    """Raised for permanent HTTP access failures that should not be retried."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Croatian documents to the existing raw/processed dataset."
    )
    parser.add_argument("--target-per-class", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation-fraction", type=float, default=0.15)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--per-document-timeout", type=int, default=120)
    parser.add_argument("--max-hrcak-pages", type=int, default=80)
    parser.add_argument(
        "--hrcak-download-approved",
        action="store_true",
        help=(
            "Confirm that Hrčak's content-mining access conditions have been met. "
            "Without this flag scientific documents use the synthetic fallback."
        ),
    )
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.target_per_class < 1:
        parser.error("--target-per-class must be positive")
    if not 0.0 <= args.augmentation_fraction <= 0.20:
        parser.error("--augmentation-fraction must be between 0 and 0.20")
    if args.request_timeout < 5 or args.per_document_timeout < 5:
        parser.error("timeouts must be at least 5 seconds")
    return args


def stable_rng(seed: int, value: str) -> random.Random:
    number = int.from_bytes(value.encode("utf-8", errors="ignore"), "little", signed=False)
    return random.Random(seed ^ number)


def font_path(bold: bool = False) -> Path | None:
    candidates = (
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    )
    return next((path for path in candidates if path.exists()), None)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = font_path(bold)
    if path:
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


class PageBuilder:
    def __init__(self, background: str = "white") -> None:
        self.image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), background)
        self.draw = ImageDraw.Draw(self.image)
        self.words: list[str] = []
        self.boxes: list[list[int]] = []
        self.text_parts: list[str] = []

    def rectangle(self, xy: Sequence[int], fill: Any, outline: Any | None = None, width: int = 1) -> None:
        self.draw.rectangle(tuple(xy), fill=fill, outline=outline, width=width)

    def line(self, xy: Sequence[int], fill: Any, width: int = 1) -> None:
        self.draw.line(tuple(xy), fill=fill, width=width)

    def _line_width(self, text: str, font: ImageFont.ImageFont) -> int:
        box = self.draw.textbbox((0, 0), text, font=font)
        return max(0, box[2] - box[0])

    def wrap(self, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        lines: list[str] = []
        for paragraph in str(text).splitlines() or [""]:
            current = ""
            for word in paragraph.split():
                proposed = f"{current} {word}".strip()
                if not current or self._line_width(proposed, font) <= max_width:
                    current = proposed
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
            elif not paragraph.strip():
                lines.append("")
        return lines

    def text(
        self,
        value: str,
        x: int,
        y: int,
        *,
        size: int = 28,
        bold: bool = False,
        fill: Any = "black",
        max_width: int | None = None,
        line_gap: int = 8,
        align: str = "left",
        record: bool = True,
    ) -> int:
        font = load_font(size, bold)
        width = max_width or (PAGE_WIDTH - x - 60)
        lines = self.wrap(value, font, width)
        if record and str(value).strip():
            self.text_parts.append(str(value).strip())
        cursor_y = y
        for line in lines:
            bbox = self.draw.textbbox((0, 0), line or "Ag", font=font)
            line_height = max(1, bbox[3] - bbox[1])
            line_width = self._line_width(line, font)
            line_x = x
            if align == "center":
                line_x = x + max(0, (width - line_width) // 2)
            elif align == "right":
                line_x = x + max(0, width - line_width)
            if line:
                word_x = line_x
                space_width = self._line_width(" ", font)
                for word in line.split():
                    word_box = self.draw.textbbox((word_x, cursor_y), word, font=font)
                    self.draw.text((word_x, cursor_y), word, font=font, fill=fill)
                    if record:
                        self.words.append(word)
                        self.boxes.append(
                            [
                                max(0, int(word_box[0])),
                                max(0, int(word_box[1])),
                                min(PAGE_WIDTH, int(word_box[2])),
                                min(PAGE_HEIGHT, int(word_box[3])),
                            ]
                        )
                    word_x = word_box[2] + space_width
            cursor_y += line_height + line_gap
        return cursor_y

    def finish(self) -> RenderedDocument:
        return RenderedDocument(
            image=self.image,
            text="\n".join(self.text_parts),
            words=self.words,
            boxes=self.boxes,
        )


def fake_oib(rng: random.Random) -> str:
    digits = [rng.randrange(10) for _ in range(10)]
    remainder = 10
    for digit in digits:
        remainder = (remainder + digit) % 10
        if remainder == 0:
            remainder = 10
        remainder = (remainder * 2) % 11
    valid_check = 11 - remainder
    if valid_check == 10:
        valid_check = 0
    invalid_check = (valid_check + rng.randrange(1, 10)) % 10
    return "".join(str(value) for value in digits) + str(invalid_check)


def fake_iban(rng: random.Random) -> str:
    return "HR00 " + " ".join(
        [
            f"{rng.randrange(1000000, 9999999):07d}",
            f"{rng.randrange(10000, 99999):05d}",
            f"{rng.randrange(1000000, 9999999):07d}",
        ]
    )


def hr_money(value: float) -> str:
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " EUR"


def synthetic_footer(builder: PageBuilder) -> None:
    builder.text(
        "Primjer dokumenta - svi podaci su izmišljeni.",
        60,
        PAGE_HEIGHT - 52,
        size=17,
        fill=(95, 95, 95),
        max_width=PAGE_WIDTH - 120,
        align="center",
    )


def render_invoice(original_id: str, seed: int) -> RenderedDocument:
    rng = stable_rng(seed, original_id)
    variants = ("klasični", "e-račun", "webshop", "usluga", "proizvod", "tablica", "bez PDV-a")
    variant = variants[rng.randrange(len(variants))]
    accents = ((19, 92, 126), (164, 47, 47), (49, 112, 79), (73, 66, 128), (40, 40, 40))
    accent = accents[rng.randrange(len(accents))]
    builder = PageBuilder("white")
    builder.rectangle((0, 0, PAGE_WIDTH, 150), accent)
    title = "E-RAČUN" if variant == "e-račun" else "RAČUN"
    builder.text(title, 62, 40, size=54, bold=True, fill="white", max_width=520)
    number = f"HR-{rng.randrange(1000, 9999)}-{rng.randrange(10, 99)}"
    builder.text(f"Broj računa: {number}", 720, 52, size=26, fill="white", max_width=450, align="right")

    company_roots = ("Primjer", "Uzorak", "Model", "Testna", "Ogledna", "Demo")
    industries = ("Trgovina", "Digital", "Usluge", "Knjiga", "Studio", "Tehnika")
    seller = f"{rng.choice(company_roots)} {rng.choice(industries)} d.o.o."
    buyer = f"Kupac {rng.choice(company_roots)} {rng.randrange(10, 999)}"
    issued = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 700))
    due = issued + timedelta(days=rng.choice((7, 14, 30)))
    left_y = 190
    left_y = builder.text("PRODAVATELJ", 62, left_y, size=22, bold=True, fill=accent, max_width=480)
    left_y = builder.text(seller, 62, left_y + 4, size=27, bold=True, max_width=480)
    left_y = builder.text("Ulica Primjera 12, 10000 Zagreb", 62, left_y + 2, size=21, max_width=480)
    left_y = builder.text(f"OIB: {fake_oib(rng)}", 62, left_y + 2, size=21, max_width=480)
    builder.text("KUPAC", 690, 190, size=22, bold=True, fill=accent, max_width=480)
    builder.text(buyer, 690, 225, size=27, bold=True, max_width=480)
    builder.text("Avenija Uzorka 5, 21000 Split", 690, 264, size=21, max_width=480)
    builder.text(f"Datum izdavanja: {issued:%d.%m.%Y.}", 690, 304, size=21, max_width=480)
    builder.text(f"Datum dospijeća: {due:%d.%m.%Y.}", 690, 338, size=21, max_width=480)

    items = (
        "Konzultantska usluga",
        "Knjiga - ogledni primjerak",
        "Uredski materijal",
        "Licenca za program",
        "Dostava proizvoda",
        "Tehnička podrška",
        "Web usluga",
        "Grafička priprema",
        "Edukacijski paket",
    )
    selected = rng.sample(items, rng.randrange(3, 7))
    table_top = 445
    builder.rectangle((55, table_top, PAGE_WIDTH - 55, table_top + 55), accent)
    headers = (("Stavka", 75), ("Kol.", 690), ("Cijena", 805), ("Iznos", 1010))
    for text, x in headers:
        builder.text(text, x, table_top + 13, size=20, bold=True, fill="white", max_width=170)
    net = 0.0
    y = table_top + 70
    for index, item in enumerate(selected, start=1):
        quantity = rng.randrange(1, 6)
        price = rng.uniform(12, 390)
        amount = quantity * price
        net += amount
        if index % 2 == 0:
            builder.rectangle((55, y - 8, PAGE_WIDTH - 55, y + 38), (243, 246, 248))
        builder.text(f"{index}. {item}", 75, y, size=20, max_width=580)
        builder.text(str(quantity), 710, y, size=20, max_width=80, align="center")
        builder.text(hr_money(price), 805, y, size=20, max_width=170, align="right")
        builder.text(hr_money(amount), 1000, y, size=20, max_width=175, align="right")
        y += 62

    vat_rate = 0 if variant == "bez PDV-a" else rng.choice((5, 13, 25))
    vat = net * vat_rate / 100
    total = net + vat
    y = max(y + 25, 980)
    builder.line((700, y, PAGE_WIDTH - 60, y), accent, 3)
    builder.text("Osnovica:", 720, y + 25, size=23, max_width=220)
    builder.text(hr_money(net), 950, y + 25, size=23, max_width=220, align="right")
    builder.text(f"PDV ({vat_rate}%):", 720, y + 67, size=23, max_width=220)
    builder.text(hr_money(vat), 950, y + 67, size=23, max_width=220, align="right")
    builder.rectangle((690, y + 115, PAGE_WIDTH - 55, y + 190), accent)
    builder.text("UKUPNO ZA PLATITI", 715, y + 134, size=25, bold=True, fill="white", max_width=300)
    builder.text(hr_money(total), 955, y + 134, size=25, bold=True, fill="white", max_width=220, align="right")
    builder.text(f"IBAN: {fake_iban(rng)}", 62, y + 230, size=22, max_width=650)
    builder.text(f"Način plaćanja: {rng.choice(('transakcijski račun', 'kartica', 'internet bankarstvo'))}", 62, y + 270, size=22, max_width=650)
    builder.text(f"Model računa: {variant}", 62, y + 310, size=18, fill=(90, 90, 90), max_width=650)
    synthetic_footer(builder)
    return builder.finish()


def render_cv(original_id: str, seed: int) -> RenderedDocument:
    rng = stable_rng(seed, original_id)
    variants = ("Europass", "moderni", "jednostavni", "dvostupčani", "studentski", "tehnički", "administrativni")
    variant = rng.choice(variants)
    palettes = ((31, 78, 121), (41, 99, 77), (112, 61, 95), (56, 56, 56), (151, 78, 42))
    accent = rng.choice(palettes)
    builder = PageBuilder((250, 250, 250) if variant == "moderni" else "white")
    sidebar = variant in {"Europass", "moderni", "dvostupčani", "tehnički"}
    if sidebar:
        builder.rectangle((0, 0, 360, PAGE_HEIGHT), accent)
    first_names = ("Ana", "Marko", "Iva", "Luka", "Petra", "Ivan", "Mia", "Nikola")
    last_names = ("Primjer", "Uzorak", "Testić", "Ogledni", "Modelić")
    name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
    role = rng.choice(
        (
            "Programer poslovnih aplikacija",
            "Administrativni suradnik",
            "Student ekonomije",
            "Grafički dizajner",
            "Projektni koordinator",
            "Stručnjak tehničke podrške",
            "Analitičar podataka",
        )
    )
    main_x = 405 if sidebar else 70
    main_width = PAGE_WIDTH - main_x - 65
    if sidebar:
        builder.text("ŽIVOTOPIS", 38, 55, size=34, bold=True, fill="white", max_width=285)
        builder.text("OSOBNI PODACI", 38, 155, size=20, bold=True, fill=(225, 235, 242), max_width=285)
        builder.text(name, 38, 195, size=27, bold=True, fill="white", max_width=285)
        builder.text("Ulica Primjera 8\n10000 Zagreb", 38, 245, size=19, fill="white", max_width=285)
        builder.text("ana.primjer@example.com", 38, 330, size=18, fill="white", max_width=285)
        builder.text("JEZICI", 38, 445, size=20, bold=True, fill=(225, 235, 242), max_width=285)
        builder.text("Hrvatski - materinski\nEngleski - B2\nNjemački - A2", 38, 485, size=19, fill="white", max_width=285)
        builder.text("DIGITALNE VJEŠTINE", 38, 640, size=20, bold=True, fill=(225, 235, 242), max_width=285)
        builder.text("Uredski alati\nTablični proračuni\nOnline suradnja\nOsnove programiranja", 38, 680, size=19, fill="white", max_width=285)
        builder.text("VOZAČKA DOZVOLA", 38, 920, size=20, bold=True, fill=(225, 235, 242), max_width=285)
        builder.text(rng.choice(("B kategorija", "AM i B kategorija", "Nema")), 38, 960, size=19, fill="white", max_width=285)
    else:
        builder.rectangle((0, 0, PAGE_WIDTH, 180), accent)
        builder.text("ŽIVOTOPIS", 70, 42, size=30, bold=True, fill="white", max_width=300)
        builder.text(name, 70, 90, size=46, bold=True, fill="white", max_width=700)
        builder.text(role, 760, 102, size=24, fill="white", max_width=400, align="right")

    y = 65 if sidebar else 235
    builder.text(name if sidebar else "PROFIL", main_x, y, size=42 if sidebar else 24, bold=True, fill=accent, max_width=main_width)
    y += 62 if sidebar else 42
    if sidebar:
        builder.text(role, main_x, y, size=27, bold=True, max_width=main_width)
        y += 52
    profile = rng.choice(
        (
            "Motivirana osoba usmjerena na kvalitetu, suradnju i odgovorno izvršavanje zadataka.",
            "Organiziran kandidat s iskustvom u radu s korisnicima, dokumentacijom i digitalnim alatima.",
            "Radoznao student zainteresiran za praktične projekte, timski rad i kontinuirano učenje.",
            "Tehnički orijentiran stručnjak koji složene probleme pretvara u jasna i održiva rješenja.",
        )
    )
    y = builder.text(profile, main_x, y, size=22, max_width=main_width, line_gap=10) + 25
    builder.text("RADNO ISKUSTVO", main_x, y, size=24, bold=True, fill=accent, max_width=main_width)
    y += 43
    experiences = rng.sample(
        (
            "2023. - 2026. | Ogledna tvrtka d.o.o. | Koordinacija zadataka i izrada izvještaja",
            "2021. - 2023. | Primjer Studio | Podrška korisnicima i vođenje evidencije",
            "2020. - 2022. | Studentski projekt | Razvoj prototipa i prezentacija rezultata",
            "2019. - 2021. | Testna ustanova | Administracija dokumenata i organizacija sastanaka",
            "2022. - 2025. | Model Digital | Analiza podataka i održavanje aplikacija",
        ),
        3,
    )
    for experience in experiences:
        y = builder.text(experience, main_x, y, size=21, max_width=main_width, line_gap=6) + 15
    y += 10
    builder.text("OBRAZOVANJE", main_x, y, size=24, bold=True, fill=accent, max_width=main_width)
    y += 43
    education = rng.choice(
        (
            "2020. - 2023. | Stručni studij računarstva | Primjer veleučilišta",
            "2019. - 2024. | Magistar ekonomije | Ogledni fakultet",
            "2022. - danas | Preddiplomski studij | Primjer sveučilišta",
            "2016. - 2020. | Tehnička škola | Smjer računalni tehničar",
        )
    )
    y = builder.text(education, main_x, y, size=21, max_width=main_width, line_gap=6) + 28
    builder.text("VJEŠTINE I PROJEKTI", main_x, y, size=24, bold=True, fill=accent, max_width=main_width)
    y += 43
    skills = rng.sample(
        (
            "Organizacija i planiranje",
            "Analiza podataka",
            "Komunikacija s korisnicima",
            "Python i SQL",
            "Upravljanje dokumentacijom",
            "Grafički alati",
            "Rad u timu",
            "Javno predstavljanje",
        ),
        5,
    )
    builder.text(" • ".join(skills), main_x, y, size=21, max_width=main_width, line_gap=8)
    builder.text(f"Stil: {variant}", main_x, PAGE_HEIGHT - 88, size=17, fill=(100, 100, 100), max_width=main_width)
    synthetic_footer(builder)
    return builder.finish()


def render_email(original_id: str, seed: int) -> RenderedDocument:
    rng = stable_rng(seed, original_id)
    variants = ("desktop", "mobile", "thread", "forward", "short", "business", "student", "link", "attachment", "signature")
    variant = rng.choice(variants)
    width = 820 if variant == "mobile" else PAGE_WIDTH
    builder = PageBuilder((246, 248, 252))
    builder.rectangle((0, 0, PAGE_WIDTH, 105), "white")
    builder.rectangle((28, 24, 78, 74), (196, 54, 47))
    builder.text("Pošta", 98, 30, size=30, bold=True, fill=(55, 55, 55), max_width=300)
    builder.text("Pretraži poruke", 370, 30, size=22, fill=(110, 110, 110), max_width=510)
    if variant != "mobile":
        builder.rectangle((0, 105, 250, PAGE_HEIGHT), (238, 242, 247))
        builder.text("Nova poruka", 30, 145, size=22, bold=True, fill=(165, 42, 42), max_width=190)
        builder.text("Primljeno\nOznačeno\nPoslano\nSkice\nArhiva", 35, 230, size=21, fill=(65, 65, 65), max_width=180, line_gap=20)
    content_x = 285 if variant != "mobile" else 65
    content_width = (PAGE_WIDTH - content_x - 65) if variant != "mobile" else width - 130
    builder.rectangle((content_x - 20, 135, content_x + content_width + 20, PAGE_HEIGHT - 80), "white", (220, 224, 230), 2)
    subjects = (
        "Dogovor za sastanak sljedeći tjedan",
        "Materijali za studentski projekt",
        "Potvrda termina radionice",
        "Izvještaj i sljedeći koraci",
        "Poveznica na dokumentaciju",
        "Molba za povratnu informaciju",
        "Obavijest o promjeni rasporeda",
        "Privitak: ogledni dokument",
    )
    subject = rng.choice(subjects)
    sender_first = rng.choice(("Ana", "Marko", "Iva", "Luka", "Petra", "Ivan"))
    sender = f"{sender_first} Primjer <{sender_first.lower()}.primjer@example.com>"
    recipient = rng.choice(("Tim projekta", "Studentska služba", "Ured za podršku", "Ogledni primatelj"))
    message_date = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 700))
    y = 175
    y = builder.text(subject, content_x, y, size=34 if variant != "mobile" else 28, bold=True, fill=(35, 35, 35), max_width=content_width) + 20
    y = builder.text(f"Pošiljatelj: {sender}", content_x, y, size=21, bold=True, max_width=content_width) + 5
    y = builder.text(f"Primatelj: {recipient} <primatelj@example.com>", content_x, y, size=19, fill=(85, 85, 85), max_width=content_width) + 5
    y = builder.text(f"Datum: {message_date:%d.%m.%Y.} u {rng.randrange(8, 18):02d}:{rng.choice((0, 15, 30, 45)):02d}", content_x, y, size=19, fill=(85, 85, 85), max_width=content_width) + 35
    greetings = ("Poštovani,", "Pozdrav svima,", "Draga kolegice,", "Dobar dan,")
    bodies = (
        "šaljem kratku potvrdu dogovora i pregled zadataka za sljedeći tjedan. Molim provjerite odgovara li predloženi termin.",
        "u privitku se nalazi ogledni dokument za naš projekt. Javite komentare kako bismo pripremili završnu verziju.",
        "materijali su dostupni na poveznici https://example.com/projekt. Poveznica vodi na demonstracijski sadržaj.",
        "hvala na poslanoj povratnoj informaciji. Izmjene su unesene, a ažurirani sažetak nalazi se u nastavku poruke.",
        "zbog promjene rasporeda sastanak se premješta na novi termin. Molim odgovorite odgovara li vam predloženo vrijeme.",
        "molim potvrdu primitka poruke. Dokumentacija je pripremljena i spremna za zajednički pregled.",
    )
    body = rng.choice(bodies)
    y = builder.text(rng.choice(greetings), content_x, y, size=23, max_width=content_width) + 15
    y = builder.text(body, content_x, y, size=23, max_width=content_width, line_gap=12) + 25
    if variant in {"thread", "forward"}:
        builder.line((content_x, y, content_x + content_width, y), (210, 210, 210), 2)
        y += 25
        prefix = "Proslijeđena poruka" if variant == "forward" else "Prethodni odgovor"
        y = builder.text(prefix, content_x, y, size=20, bold=True, fill=(90, 90, 90), max_width=content_width) + 10
        y = builder.text("Hvala na obavijesti. Pregledat ću sadržaj i odgovoriti do kraja dana.", content_x, y, size=20, fill=(95, 95, 95), max_width=content_width) + 25
    if variant == "attachment":
        builder.rectangle((content_x, y, content_x + 360, y + 74), (242, 245, 248), (190, 195, 202), 2)
        builder.text("Privitak: primjer_dokumenta.pdf", content_x + 20, y + 20, size=19, bold=True, max_width=320)
        y += 105
    builder.text(rng.choice(("Lijep pozdrav,", "Srdačan pozdrav,", "Hvala i lijep pozdrav,")), content_x, y, size=22, max_width=content_width)
    builder.text(sender_first + " Primjer", content_x, y + 42, size=22, bold=True, max_width=content_width)
    builder.text("Odgovori    Proslijedi", content_x, PAGE_HEIGHT - 155, size=20, bold=True, fill=(55, 91, 166), max_width=400)
    synthetic_footer(builder)
    return builder.finish()


def sanitized_contract_fields(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_key, raw_value in record.items():
        key = re.sub(r"\s+", " ", str(raw_key or "")).strip()
        value = re.sub(r"\s+", " ", str(raw_value or "")).strip()
        normalized_key = normalize_text(key)
        if not key or not value or value.casefold() in {"nan", "none", "null", "-"}:
            continue
        if any(marker in normalized_key for marker in SENSITIVE_FIELD_MARKERS):
            continue
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        result.append((key[:80], value[:350]))
        if len(result) >= 12:
            break
    return result


def render_contract_record(candidate: Candidate, seed: int) -> RenderedDocument:
    rng = stable_rng(seed, candidate.original_id)
    builder = PageBuilder("white")
    accents = ((41, 78, 108), (90, 80, 54), (48, 94, 72), (91, 58, 90))
    accent = rng.choice(accents)
    builder.rectangle((0, 0, PAGE_WIDTH, 170), accent)
    builder.text("REGISTAR UGOVORA", 65, 42, size=44, bold=True, fill="white", max_width=750)
    builder.text("Javna nabava", 820, 65, size=25, fill="white", max_width=350, align="right")
    organization = str(candidate.data.get("organization", "Javna institucija"))
    dataset_title = str(candidate.data.get("dataset_title", "Javni registar ugovora"))
    builder.text(organization, 65, 210, size=31, bold=True, fill=accent, max_width=1110)
    builder.text(dataset_title, 65, 260, size=22, fill=(75, 75, 75), max_width=1110)
    fields = sanitized_contract_fields(candidate.data.get("record", {}))
    y = 340
    for index, (key, value) in enumerate(fields):
        row_height = 78 if len(value) < 90 else 112
        if index % 2 == 0:
            builder.rectangle((55, y - 10, PAGE_WIDTH - 55, y + row_height - 12), (244, 246, 248))
        builder.text(key, 75, y, size=20, bold=True, fill=accent, max_width=330)
        builder.text(value, 410, y, size=20, max_width=740, line_gap=5)
        y += row_height
        if y > PAGE_HEIGHT - 160:
            break
    builder.line((55, PAGE_HEIGHT - 120, PAGE_WIDTH - 55, PAGE_HEIGHT - 120), accent, 2)
    builder.text("Izvor: javno dostupni podaci hrvatskog javnog sektora", 65, PAGE_HEIGHT - 95, size=18, fill=(90, 90, 90), max_width=1100)
    return builder.finish()


def render_synthetic_contract(original_id: str, seed: int) -> RenderedDocument:
    rng = stable_rng(seed, original_id)
    record = {
        "Evidencijski broj nabave": f"JN-{rng.randrange(100, 999)}-{rng.randrange(10, 99)}",
        "Predmet ugovora": rng.choice(
            (
                "Održavanje informatičke opreme",
                "Nabava uredskog materijala",
                "Usluga stručnog savjetovanja",
                "Radovi na uređenju prostora",
                "Nabava edukacijskih materijala",
            )
        ),
        "Vrsta postupka": rng.choice(("Otvoreni postupak", "Jednostavna nabava", "Okvirni sporazum")),
        "Datum sklapanja": f"{rng.randrange(1, 28):02d}.{rng.randrange(1, 13):02d}.2026.",
        "Vrijednost ugovora bez PDV-a": hr_money(rng.uniform(2500, 95000)),
        "Rok izvršenja": rng.choice(("30 dana", "90 dana", "12 mjeseci", "do završetka projekta")),
        "Status": rng.choice(("U izvršenju", "Izvršen", "Aktivan")),
        "Napomena": "Primjer zapisa s izmišljenim podacima za istraživačku uporabu.",
    }
    candidate = Candidate(
        label="contract",
        source_name="hr_synthetic_contract",
        source_locator="generated://croatian-public-procurement-example",
        original_id=original_id,
        kind="contract_record",
        is_synthetic=True,
        data={
            "organization": f"Ogledna javna ustanova {rng.randrange(1, 99)}",
            "dataset_title": "Primjer registra ugovora i okvirnih sporazuma",
            "record": record,
        },
    )
    rendered = render_contract_record(candidate, seed)
    builder = PageBuilder()
    builder.image = rendered.image
    builder.draw = ImageDraw.Draw(builder.image)
    builder.words = rendered.words
    builder.boxes = rendered.boxes
    builder.text_parts = rendered.text.splitlines()
    synthetic_footer(builder)
    return builder.finish()


def render_synthetic_scientific(original_id: str, seed: int) -> RenderedDocument:
    rng = stable_rng(seed, original_id)
    backgrounds = ("white", (249, 250, 248), (247, 249, 252))
    builder = PageBuilder(rng.choice(backgrounds))
    accents = ((30, 74, 103), (72, 88, 55), (113, 65, 81), (73, 69, 128), (45, 95, 91))
    accent = rng.choice(accents)
    layout = rng.choice(("top_band", "left_rule", "abstract_box", "journal_rule"))
    if layout == "top_band":
        builder.rectangle((0, 0, PAGE_WIDTH, 34), accent)
    elif layout == "left_rule":
        builder.rectangle((0, 0, 24, PAGE_HEIGHT), accent)
    elif layout == "journal_rule":
        builder.line((70, 98, PAGE_WIDTH - 70, 98), accent, 5)

    fields = (
        "informacijskih znanosti",
        "obrazovanja",
        "održivog razvoja",
        "ekonomije",
        "jezikoslovlja",
        "javne uprave",
        "prometnih sustava",
        "kulturologije",
        "zdravstvene informatike",
        "turističkog menadžmenta",
        "urbane geografije",
        "socijalne pedagogije",
    )
    topics = (
        "digitalnih praksi",
        "otvorenih podataka",
        "korisničkog iskustva",
        "suradničkog učenja",
        "lokalnih razvojnih mjera",
        "stručne komunikacije",
        "energetske učinkovitosti",
        "dostupnosti javnih usluga",
        "informacijske pismenosti",
        "organizacijske otpornosti",
        "mobilnosti stanovništva",
        "upravljanja projektnim rizicima",
    )
    contexts = (
        "hrvatskim gradovima",
        "visokom obrazovanju",
        "malim organizacijama",
        "lokalnim zajednicama",
        "digitalnim knjižnicama",
        "javnim ustanovama",
        "regionalnim poduzećima",
        "studentskim projektima",
        "kulturnim institucijama",
        "mrežnim uslugama",
    )
    methods = (
        "deskriptivna statistika i analiza sadržaja",
        "usporedna studija slučaja",
        "tematsko kodiranje strukturiranih zapisa",
        "višekriterijska procjena pokazatelja",
        "longitudinalno praćenje promjena",
        "analiza mreža i polustrukturirani upitnik",
        "eksperimentalna usporedba triju scenarija",
        "kombinirana kvalitativna i kvantitativna metoda",
    )
    field_name = rng.choice(fields)
    topic = rng.choice(topics)
    context = rng.choice(contexts)
    method = rng.choice(methods)
    title_opening = rng.choice(("Analiza", "Procjena", "Modeliranje", "Vrednovanje", "Istraživanje"))
    title = f"{title_opening} {topic} u {context}"
    sample_size = rng.randrange(48, 960)
    indicator_count = rng.randrange(6, 24)
    improvement = rng.randrange(7, 48)
    study_year = rng.randrange(2018, 2027)

    builder.text("OGLEDNI HRVATSKI ZNANSTVENI RAD", 70, 52, size=18, bold=True, fill=(85, 85, 85), max_width=1100, align="center")
    builder.text(title, 85, 112, size=36, bold=True, fill=accent, max_width=1070, align="center", line_gap=12)
    builder.text(
        f"Autor Primjer {rng.randrange(10, 999)} | Ogledna istraživačka ustanova",
        90,
        245,
        size=20,
        fill=(80, 80, 80),
        max_width=1060,
        align="center",
    )
    y = 325
    if layout == "abstract_box":
        builder.rectangle((58, y - 18, PAGE_WIDTH - 58, y + 245), (237, 241, 244), accent, 2)
    builder.text("Sažetak", 80, y, size=26, bold=True, fill=accent, max_width=1080)
    abstract = (
        f"Rad istražuje {topic} u području {field_name}. Uzorak obuhvaća {sample_size} "
        f"anonimnih sintetičkih zapisa iz {study_year}. godine. Primijenjena je metoda: "
        f"{method}. Analiza {indicator_count} pokazatelja pokazala je poboljšanje od "
        f"{improvement} posto u odabranom scenariju, uz ograničenja povezana s veličinom "
        "uzorka i lokalnim kontekstom."
    )
    y = builder.text(abstract, 80, y + 42, size=21, max_width=1080, line_gap=9) + 16
    keywords = rng.sample(topics, 3)
    y = builder.text(
        "Ključne riječi: " + "; ".join(keywords),
        80,
        y,
        size=18,
        bold=True,
        fill=(75, 75, 75),
        max_width=1080,
    ) + 25

    introduction = rng.choice(
        (
            f"U posljednjem desetljeću {topic} postaje važan čimbenik planiranja u {context}. Literatura naglašava dostupnost, pouzdanost i transparentnost postupaka.",
            f"Promjene povezane s područjem {field_name} potiču razvoj novih načina mjerenja. Posebna pozornost posvećena je razlikama između institucionalnih okruženja.",
            f"Dosadašnja istraživanja {topic} daju neujednačene nalaze. Ovaj rad zato uvodi jasno definirane pokazatelje i ponovljiv postupak usporedbe.",
        )
    )
    methodology = rng.choice(
        (
            f"Korištena je {method}. Zapisi su podijeljeni u {rng.randrange(3, 8)} skupina, a kvaliteta je provjerena dvostrukim kodiranjem i analizom odstupanja.",
            f"Istraživački nacrt povezuje {indicator_count} pokazatelja s trima razinama procjene. Podaci su standardizirani, provjereni i obrađeni bez stvarnih osobnih podataka.",
            f"Postupak uključuje pripremu uzorka, kontrolu nedostajućih vrijednosti i {method}. Stabilnost nalaza provjerena je u {rng.randrange(4, 12)} ponavljanja.",
        )
    )
    results = rng.choice(
        (
            f"Najveća promjena zabilježena je za pokazatelj dostupnosti ({improvement} posto), dok je najmanja iznosila {rng.randrange(2, 12)} posto. Razlike su bile stabilne u većini skupina.",
            f"Od {indicator_count} pokazatelja, njih {rng.randrange(3, indicator_count)} ostvarilo je vrijednost iznad unaprijed definiranog praga. Rezultat upućuje na povezanost konteksta i načina primjene.",
            f"Usporedba scenarija pokazuje raspon rezultata od {rng.randrange(31, 55)} do {rng.randrange(72, 94)} bodova. Kvalitativni zapisi dodatno objašnjavaju uočene razlike.",
        )
    )
    discussion = rng.choice(
        (
            "Nalazi podupiru potrebu za jasnim pravilima prikupljanja podataka i redovitom provjerom mjernih instrumenata. Generalizacija je ograničena na opisani kontekst.",
            "Rezultate treba promatrati zajedno s organizacijskim i vremenskim čimbenicima. Budući rad može uključiti više regija i dulje razdoblje opažanja.",
            "Uočene razlike potvrđuju da jedan pokazatelj nije dovoljan za pouzdanu procjenu. Kombiniranje više izvora daje stabilniji i interpretabilniji zaključak.",
        )
    )
    conclusion = rng.choice(
        (
            f"Predloženi postupak prikladan je za početno vrednovanje {topic}. Daljnja provjera treba uključiti neovisne izvore i unaprijed registriran plan analize.",
            f"Istraživanje pokazuje praktičnu vrijednost sustavnog praćenja u {context}. Rezultati otvaraju pitanja o prijenosu metode na druga područja.",
            f"Kombinacija pokazatelja i kvalitativnog tumačenja poboljšava razumijevanje {topic}. Potrebna su nova istraživanja s raznovrsnijim uzorkom.",
        )
    )
    for section, content in (
        ("1. Uvod", introduction),
        ("2. Metodologija", methodology),
        ("3. Rezultati", results),
        ("4. Rasprava", discussion),
        ("5. Zaključak", conclusion),
    ):
        builder.text(section, 80, y, size=24, bold=True, fill=accent, max_width=1080)
        y = builder.text(content, 80, y + 36, size=20, max_width=1080, line_gap=8) + 17
        if y > PAGE_HEIGHT - 300:
            break

    chart_y = min(y + 5, PAGE_HEIGHT - 230)
    builder.line((80, chart_y + 112, 520, chart_y + 112), (110, 110, 110), 2)
    for index in range(4):
        height = rng.randrange(35, 105)
        x = 110 + index * 95
        builder.rectangle((x, chart_y + 110 - height, x + 54, chart_y + 110), accent)
    builder.text(
        f"Slika 1. Sažeti prikaz četiriju pokazatelja, oznaka studije {rng.randrange(1000, 9999)}.",
        560,
        chart_y + 42,
        size=17,
        fill=(85, 85, 85),
        max_width=570,
    )
    synthetic_footer(builder)
    return builder.finish()


def sanitize_public_value(value: Any) -> str:
    """Remove contact identifiers while retaining procurement facts."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[kontakt uklonjen]", text)
    text = re.sub(
        r"\bHR\d{2}(?:[\s-]*\d){17,21}\b",
        "[IBAN uklonjen]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<!\d)\d{11}(?!\d)", "[identifikator uklonjen]", text)
    text = re.sub(
        r"(?<!\d)(?:\+?385|0)[\s./-]?(?:\d[\s./-]?){8,9}(?!\d)",
        "[telefon uklonjen]",
        text,
    )
    return text[:500]


class HttpClient:
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})

    def close(self) -> None:
        self.session.close()

    def get_bytes(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        max_bytes: int = MAX_DOWNLOAD_BYTES,
    ) -> tuple[bytes, str]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with self.session.get(
                    url,
                    params=params,
                    timeout=(15, self.timeout),
                    stream=True,
                    allow_redirects=True,
                ) as response:
                    if response.status_code in {401, 403, 418}:
                        raise RemoteAccessDenied(
                            f"HTTP {response.status_code} access denied for {response.url}"
                        )
                    response.raise_for_status()
                    declared = int(response.headers.get("Content-Length", "0") or 0)
                    if declared > max_bytes:
                        raise DatasetExpansionError(
                            f"Download is too large ({declared} bytes; limit {max_bytes}): {url}"
                        )
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise DatasetExpansionError(
                                f"Download exceeded {max_bytes} bytes: {url}"
                            )
                        chunks.append(chunk)
                    return b"".join(chunks), response.url
            except RemoteAccessDenied:
                raise
            except (requests.RequestException, DatasetExpansionError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise DatasetExpansionError(f"Download failed for {url}: {last_error}")

    def get_json(self, url: str, *, params: Mapping[str, Any]) -> Mapping[str, Any]:
        payload, _ = self.get_bytes(url, params=params, max_bytes=20 * 1024 * 1024)
        result = json.loads(payload.decode("utf-8-sig"))
        if not isinstance(result, Mapping):
            raise DatasetExpansionError(f"Expected a JSON object from {url}")
        return result


def add_duplicate_report(
    rows: list[dict[str, object]],
    *,
    candidate_id: str,
    label: str,
    source_name: str,
    decision: str,
    reason: str,
    candidate_path: str = "",
    similar_to: str = "",
    similarity_score: float | str = "",
) -> None:
    rows.append(
        {
            "candidate_id": candidate_id,
            "candidate_path": candidate_path,
            "label": label,
            "source_name": source_name,
            "decision": decision,
            "reason": reason,
            "similar_to": similar_to,
            "similarity_score": (
                f"{similarity_score:.6f}" if isinstance(similarity_score, float) else similarity_score
            ),
        }
    )


def _xml_values(parent: ET.Element, local_name: str) -> list[str]:
    result: list[str] = []
    for element in parent.iter():
        if element.tag.rsplit("}", 1)[-1].casefold() == local_name.casefold() and element.text:
            value = re.sub(r"\s+", " ", element.text).strip()
            if value:
                result.append(value)
    return result


def _is_croatian_language(values: Iterable[str]) -> bool:
    for raw_value in values:
        value = normalize_text(raw_value).replace("_", "-")
        if value in {"hr", "hrv", "cro", "hr-hr", "hrvatski", "croatian"}:
            return True
        if value.startswith("hrvats"):
            return True
    return False


def round_robin_groups(
    groups: Mapping[str, Sequence[Candidate]], limit: int
) -> list[Candidate]:
    queues = [(key, deque(values)) for key, values in sorted(groups.items()) if values]
    result: list[Candidate] = []
    while queues and len(result) < limit:
        next_round: list[tuple[str, deque[Candidate]]] = []
        for key, queue in queues:
            if queue and len(result) < limit:
                result.append(queue.popleft())
            if queue:
                next_round.append((key, queue))
        queues = next_round
    return result


def collect_hrcak_candidates(
    client: HttpClient,
    needed: int,
    max_pages: int,
    report_rows: list[dict[str, object]],
) -> list[Candidate]:
    """Collect Croatian-language PDF metadata through Hrčak's official OAI endpoint."""
    if needed <= 0:
        return []
    target_pool = max(needed * 4, needed + 80)
    by_journal: dict[str, list[Candidate]] = defaultdict(list)
    seen_urls: set[str] = set()
    token = ""

    print("Dohvaćam hrvatske znanstvene radove preko Hrčak OAI sučelja...")
    for page_number in range(1, max_pages + 1):
        params: dict[str, str] = {"verb": "ListRecords"}
        if token:
            params["resumptionToken"] = token
        else:
            params["metadataPrefix"] = "oai_dc"
        try:
            payload, _ = client.get_bytes(HRCAK_OAI_URL, params=params, max_bytes=25 * 1024 * 1024)
            root = ET.fromstring(payload)
        except Exception as error:
            add_duplicate_report(
                report_rows,
                candidate_id=f"hrcak_oai_page_{page_number}",
                label="scientific",
                source_name="hrcak",
                decision="skipped",
                reason=f"source_listing_error: {error}",
            )
            break

        for record in root.iter():
            if record.tag.rsplit("}", 1)[-1] != "record":
                continue
            metadata = next(
                (child for child in record if child.tag.rsplit("}", 1)[-1] == "metadata"),
                None,
            )
            if metadata is None or not _is_croatian_language(_xml_values(metadata, "language")):
                continue
            identifiers = _xml_values(metadata, "identifier")
            pdf_url = next(
                (
                    value
                    for value in identifiers
                    if value.casefold().startswith(("http://", "https://"))
                    and "/file/" in value.casefold()
                ),
                "",
            )
            if not pdf_url or pdf_url in seen_urls:
                continue
            titles = _xml_values(metadata, "title")
            sources = _xml_values(metadata, "source")
            header_ids = _xml_values(record, "identifier")
            journal = sources[0] if sources else "Hrčak - ostali časopisi"
            original_id = header_ids[0] if header_ids else pdf_url
            seen_urls.add(pdf_url)
            by_journal[journal].append(
                Candidate(
                    label="scientific",
                    source_name="hrcak",
                    source_locator=pdf_url,
                    original_id=original_id,
                    kind="download_pdf",
                    is_synthetic=False,
                    data={
                        "title": titles[0] if titles else "Znanstveni rad",
                        "journal": journal,
                    },
                    extension=".pdf",
                )
            )

        candidates_found = sum(len(items) for items in by_journal.values())
        if candidates_found >= target_pool:
            break
        token_element = next(
            (
                element
                for element in root.iter()
                if element.tag.rsplit("}", 1)[-1] == "resumptionToken"
            ),
            None,
        )
        token = (token_element.text or "").strip() if token_element is not None else ""
        if not token:
            break

    balanced = round_robin_groups(by_journal, target_pool)
    print(
        f"Hrčak kandidati: {len(balanced)} iz {len(by_journal)} različitih časopisa."
    )
    return balanced


def decode_text_payload(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1250", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def rows_from_csv(payload: bytes) -> list[dict[str, Any]]:
    text = decode_text_payload(payload)
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
        dialect.delimiter = ";"
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    return [dict(row) for row in reader if any(str(value or "").strip() for value in row.values())]


def _largest_record_list(value: Any) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    if isinstance(value, list):
        mapped = [dict(item) for item in value if isinstance(item, Mapping)]
        if len(mapped) > len(best):
            best = mapped
        for item in value:
            nested = _largest_record_list(item)
            if len(nested) > len(best):
                best = nested
    elif isinstance(value, Mapping):
        for item in value.values():
            nested = _largest_record_list(item)
            if len(nested) > len(best):
                best = nested
    return best


def rows_from_json(payload: bytes) -> list[dict[str, Any]]:
    return _largest_record_list(json.loads(payload.decode("utf-8-sig")))


def excel_column_index(reference: str) -> int:
    letters = re.match(r"[A-Za-z]+", reference)
    result = 0
    for character in (letters.group(0).upper() if letters else "A"):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def rows_from_xlsx(payload: bytes) -> list[dict[str, Any]]:
    """Read simple XLSX tables with stdlib only; no extra deployment dependency."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.iter():
                if item.tag.rsplit("}", 1)[-1] == "si":
                    shared.append("".join(node.text or "" for node in item.iter() if node.tag.rsplit("}", 1)[-1] == "t"))

        all_records: list[dict[str, Any]] = []
        sheet_names = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        )
        for sheet_name in sheet_names:
            root = ET.fromstring(archive.read(sheet_name))
            table: list[list[str]] = []
            for row_element in root.iter():
                if row_element.tag.rsplit("}", 1)[-1] != "row":
                    continue
                row: list[str] = []
                for cell in row_element:
                    if cell.tag.rsplit("}", 1)[-1] != "c":
                        continue
                    index = excel_column_index(cell.attrib.get("r", "A1"))
                    while len(row) <= index:
                        row.append("")
                    cell_type = cell.attrib.get("t", "")
                    raw_value = ""
                    if cell_type == "inlineStr":
                        raw_value = "".join(
                            node.text or ""
                            for node in cell.iter()
                            if node.tag.rsplit("}", 1)[-1] == "t"
                        )
                    else:
                        value_element = next(
                            (node for node in cell if node.tag.rsplit("}", 1)[-1] == "v"),
                            None,
                        )
                        raw_value = (value_element.text or "") if value_element is not None else ""
                        if cell_type == "s" and raw_value.isdigit():
                            shared_index = int(raw_value)
                            raw_value = shared[shared_index] if shared_index < len(shared) else raw_value
                    row[index] = raw_value
                if any(value.strip() for value in row):
                    table.append(row)
            if len(table) < 2:
                continue
            headers = [re.sub(r"\s+", " ", value).strip() or f"column_{index + 1}" for index, value in enumerate(table[0])]
            for values in table[1:]:
                record = {
                    header: values[index] if index < len(values) else ""
                    for index, header in enumerate(headers)
                }
                if any(str(value).strip() for value in record.values()):
                    all_records.append(record)
        return all_records


def parse_tabular_resource(payload: bytes, resource_format: str, url: str) -> list[dict[str, Any]]:
    format_name = resource_format.casefold().strip()
    suffix = Path(urllib.parse.urlparse(url).path).suffix.casefold()
    if "csv" in format_name or suffix == ".csv":
        return rows_from_csv(payload)
    if "json" in format_name or suffix == ".json":
        return rows_from_json(payload)
    if "xlsx" in format_name or suffix == ".xlsx":
        return rows_from_xlsx(payload)
    return []


def collect_contract_candidates(
    client: HttpClient,
    needed: int,
    report_rows: list[dict[str, object]],
) -> list[Candidate]:
    if needed <= 0:
        return []
    queries = (
        "registar ugovora javna nabava",
        "javna nabava ugovori",
        "okvirni sporazum",
    )
    resources: dict[str, dict[str, str]] = {}
    for query in queries:
        try:
            response = client.get_json(DATA_GOV_API_URL, params={"q": query, "rows": 100})
            packages = response.get("result", {}).get("results", [])  # type: ignore[union-attr]
        except Exception as error:
            add_duplicate_report(
                report_rows,
                candidate_id=f"data_gov_query:{query}",
                label="contract",
                source_name="hrvatska_javna_nabava",
                decision="skipped",
                reason=f"source_listing_error: {error}",
            )
            continue
        for package in packages if isinstance(packages, list) else []:
            if not isinstance(package, Mapping):
                continue
            organization_data = package.get("organization")
            organization = (
                str(organization_data.get("title", ""))
                if isinstance(organization_data, Mapping)
                else "Javna institucija"
            )
            dataset_title = str(package.get("title", "Javni registar ugovora"))
            package_id = str(package.get("id", package.get("name", "dataset")))
            for resource in package.get("resources", []) if isinstance(package.get("resources"), list) else []:
                if not isinstance(resource, Mapping):
                    continue
                url = str(resource.get("url", "")).strip()
                resource_format = str(resource.get("format", "")).strip()
                suffix = Path(urllib.parse.urlparse(url).path).suffix.casefold()
                if not url or not (
                    any(name in resource_format.casefold() for name in ("csv", "json", "xlsx"))
                    or suffix in {".csv", ".json", ".xlsx"}
                ):
                    continue
                resource_id = str(resource.get("id", url))
                resources[resource_id] = {
                    "url": url,
                    "format": resource_format,
                    "organization": organization or "Javna institucija",
                    "dataset_title": dataset_title,
                    "package_id": package_id,
                    "resource_id": resource_id,
                }

    pools: dict[str, list[Candidate]] = defaultdict(list)
    print(f"Pronađeno tabličnih data.gov.hr resursa: {len(resources)}")
    for resource in list(resources.values())[:80]:
        if sum(len(items) for items in pools.values()) >= max(needed * 3, needed + 60):
            break
        try:
            payload, final_url = client.get_bytes(resource["url"], max_bytes=30 * 1024 * 1024)
            records = parse_tabular_resource(payload, resource["format"], final_url)
        except Exception as error:
            add_duplicate_report(
                report_rows,
                candidate_id=resource["resource_id"],
                label="contract",
                source_name="hrvatska_javna_nabava",
                decision="skipped",
                reason=f"resource_parse_error: {error}",
                candidate_path=resource["url"],
            )
            continue
        accepted_from_resource = 0
        for row_index, record in enumerate(records):
            fields = sanitized_contract_fields(record)
            if len(fields) < 3:
                continue
            sanitized_record = {key: sanitize_public_value(value) for key, value in fields}
            digest = hashlib.sha1(
                f"{resource['resource_id']}:{row_index}".encode("utf-8")
            ).hexdigest()[:16]
            pools[resource["organization"]].append(
                Candidate(
                    label="contract",
                    source_name="hrvatska_javna_nabava",
                    source_locator=final_url,
                    original_id=f"data-gov-hr:{digest}",
                    kind="contract_record",
                    is_synthetic=False,
                    data={
                        "organization": sanitize_public_value(resource["organization"]),
                        "dataset_title": sanitize_public_value(resource["dataset_title"]),
                        "record": sanitized_record,
                    },
                    extension=".pdf",
                )
            )
            accepted_from_resource += 1
            if accepted_from_resource >= 12 or len(pools[resource["organization"]]) >= 30:
                break
    balanced = round_robin_groups(pools, max(needed * 2, needed + 30))
    print(
        f"Kandidati javne nabave: {len(balanced)} iz {len(pools)} različitih organizacija."
    )
    return balanced


def synthetic_candidates(label: str, seed: int, start: int = 1) -> Iterator[Candidate]:
    index = start
    while True:
        original_id = f"generated:{label}:hr:{seed}:{index:06d}"
        if label == "invoice":
            source_name, kind, extension = "hr_synthetic_invoice", "invoice", ".pdf"
        elif label == "cv":
            source_name, kind, extension = "hr_synthetic_cv", "cv", ".pdf"
        elif label == "email":
            source_name, kind, extension = "hr_synthetic_gmail_like", "email", ".png"
        elif label == "contract":
            source_name, kind, extension = "hr_synthetic_contract", "synthetic_contract", ".pdf"
        elif label == "scientific":
            source_name, kind, extension = "hr_synthetic_scientific", "synthetic_scientific", ".pdf"
        else:
            raise ValueError(f"Unknown class: {label}")
        yield Candidate(
            label=label,
            source_name=source_name,
            source_locator=f"generated://croatian/{label}",
            original_id=original_id,
            kind=kind,
            is_synthetic=True,
            extension=extension,
        )
        index += 1


def next_synthetic_original_index(
    label: str,
    seed: int,
    existing_source_rows: Iterable[Mapping[str, str]],
) -> int:
    pattern = re.compile(
        rf"^generated:{re.escape(label)}:hr:{seed}:(\d+)$"
    )
    numbers = [
        int(match.group(1))
        for row in existing_source_rows
        if not str(row.get("is_augmented", "")).casefold() in {"true", "1", "yes"}
        and (match := pattern.fullmatch(str(row.get("original_id", ""))))
    ]
    return max(numbers, default=0) + 1


def render_candidate(candidate: Candidate, seed: int) -> RenderedDocument:
    if candidate.kind == "invoice":
        rendered = render_invoice(candidate.original_id, seed)
    elif candidate.kind == "cv":
        rendered = render_cv(candidate.original_id, seed)
    elif candidate.kind == "email":
        rendered = render_email(candidate.original_id, seed)
    elif candidate.kind == "contract_record":
        rendered = render_contract_record(candidate, seed)
    elif candidate.kind == "synthetic_contract":
        rendered = render_synthetic_contract(candidate.original_id, seed)
    elif candidate.kind == "synthetic_scientific":
        rendered = render_synthetic_scientific(candidate.original_id, seed)
    else:
        raise ValueError(f"Candidate kind is not rendered: {candidate.kind}")
    if not rendered.words or len(rendered.words) != len(rendered.boxes):
        raise DatasetExpansionError(f"Invalid rendered OCR data for {candidate.original_id}")
    return rendered


def staged_paths(staging_root: Path, label: str, document_id: str, extension: str) -> tuple[Path, Path, Path, Path]:
    return (
        staging_root / "raw" / label / f"{document_id}{extension}",
        staging_root / "processed" / "images" / f"{document_id}.png",
        staging_root / "processed" / "texts" / f"{document_id}.txt",
        staging_root / "processed" / "ocr" / f"{document_id}.json",
    )


def final_paths(label: str, document_id: str, extension: str) -> tuple[Path, Path, Path, Path]:
    return (
        RAW_DIR / label / f"{document_id}{extension}",
        PROCESSED_DIR / "images" / f"{document_id}.png",
        PROCESSED_DIR / "texts" / f"{document_id}.txt",
        PROCESSED_DIR / "ocr" / f"{document_id}.json",
    )


def write_rendered_outputs(
    rendered: RenderedDocument,
    *,
    label: str,
    raw_path: Path,
    image_path: Path,
    text_path: Path,
    ocr_path: Path,
) -> None:
    for path in (raw_path, image_path, text_path, ocr_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    image = rendered.image.convert("RGB")
    if raw_path.suffix.casefold() == ".pdf":
        image.save(raw_path, format="PDF", resolution=150.0)
    else:
        image.save(raw_path, format="PNG", optimize=True)
    image.save(image_path, format="PNG", optimize=True)
    text_path.write_text(rendered.text.strip(), encoding="utf-8")
    payload = {
        "words": rendered.words,
        "boxes": rendered.boxes,
        "label": label,
        "confidences": ["100"] * len(rendered.words),
        "page_indices": [0] * len(rendered.words),
        "image_width": image.width,
        "image_height": image.height,
    }
    ocr_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def processed_outputs_valid(image_path: Path, text_path: Path, ocr_path: Path) -> tuple[bool, str]:
    missing = [str(path) for path in (image_path, text_path, ocr_path) if not path.exists()]
    if missing:
        return False, f"missing output: {', '.join(missing)}"
    try:
        with Image.open(image_path) as image:
            image.verify()
        text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(normalize_text(text)) < 20:
            return False, "processed text has fewer than 20 normalized characters"
        payload = json.loads(ocr_path.read_text(encoding="utf-8"))
        words = payload.get("words", [])
        boxes = payload.get("boxes", [])
        if not words or len(words) != len(boxes):
            return False, "OCR words/boxes are empty or mismatched"
        if any(not isinstance(box, list) or len(box) != 4 for box in boxes):
            return False, "OCR contains an invalid bounding box"
    except Exception as error:
        return False, str(error)
    return True, ""


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


def run_preprocess_worker(job_path: Path, timeout_seconds: int) -> tuple[bool, str, float]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-job", str(job_path)]
    kwargs: dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    started = time.monotonic()
    process = subprocess.Popen(command, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_worker_tree(process)
        return False, f"preprocessing timeout after {timeout_seconds} seconds", time.monotonic() - started
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        detail = (stderr or stdout or f"worker exited with code {process.returncode}").strip()
        return False, detail[-4000:], elapsed
    return True, "", elapsed


def run_worker_job(job_path: Path) -> None:
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
    process_file_to_outputs(raw_path, str(job["label"]), image_path, text_path, ocr_path)
    valid, reason = processed_outputs_valid(image_path, text_path, ocr_path)
    if not valid:
        raise RuntimeError(reason)


def valid_pdf_payload(payload: bytes) -> bool:
    if not payload.lstrip().startswith(b"%PDF"):
        return False
    try:
        with fitz.open(stream=payload, filetype="pdf") as document:
            return document.page_count > 0
    except Exception:
        return False


def prepare_candidate(
    candidate: Candidate,
    document_id: str,
    staging_root: Path,
    client: HttpClient,
    args: argparse.Namespace,
) -> PreparedDocument:
    extension = candidate.extension.casefold()
    raw_stage, image_stage, text_stage, ocr_stage = staged_paths(
        staging_root, candidate.label, document_id, extension
    )
    raw_final, _, _, _ = final_paths(candidate.label, document_id, extension)
    if candidate.kind == "download_pdf":
        payload = candidate.payload
        final_url = candidate.source_locator
        if payload is None:
            payload, final_url = client.get_bytes(candidate.source_locator)
        if not valid_pdf_payload(payload):
            raise DatasetExpansionError("downloaded content is not a readable PDF")
        raw_stage.parent.mkdir(parents=True, exist_ok=True)
        raw_stage.write_bytes(payload)
        jobs_dir = staging_root / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job_path = jobs_dir / f"{document_id}.json"
        job_path.write_text(
            json.dumps(
                {
                    "raw_path": str(raw_stage),
                    "label": candidate.label,
                    "image_path": str(image_stage),
                    "text_path": str(text_stage),
                    "ocr_path": str(ocr_stage),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        success, reason, _ = run_preprocess_worker(job_path, args.per_document_timeout)
        if not success:
            raise DatasetExpansionError(reason)
        candidate.source_locator = final_url
    else:
        rendered = render_candidate(candidate, args.seed)
        write_rendered_outputs(
            rendered,
            label=candidate.label,
            raw_path=raw_stage,
            image_path=image_stage,
            text_path=text_stage,
            ocr_path=ocr_stage,
        )
    valid, reason = processed_outputs_valid(image_stage, text_stage, ocr_stage)
    if not valid:
        raise DatasetExpansionError(reason)
    return PreparedDocument(
        document_id=document_id,
        label=candidate.label,
        source_name=candidate.source_name,
        source_locator=candidate.source_locator,
        original_id=candidate.original_id,
        language="hr",
        is_synthetic=candidate.is_synthetic,
        is_augmented=False,
        augmentation_type="",
        parent_id="",
        raw_stage_path=raw_stage,
        image_stage_path=image_stage,
        text_stage_path=text_stage,
        ocr_stage_path=ocr_stage,
        raw_final_path=raw_final,
    )


def apply_augmentation(image: Image.Image, augmentation_type: str, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    if augmentation_type == "brightness_low":
        return ImageEnhance.Brightness(image).enhance(rng.uniform(0.62, 0.82))
    if augmentation_type == "slight_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.55, 1.15)))
    if augmentation_type == "low_contrast":
        return ImageEnhance.Contrast(image).enhance(rng.uniform(0.62, 0.82))
    if augmentation_type == "jpeg_compression":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randrange(38, 62), optimize=True)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    if augmentation_type == "screenshot_crop":
        width, height = image.size
        left = rng.randrange(8, max(9, min(45, width // 20)))
        top = rng.randrange(8, max(9, min(45, height // 20)))
        right = width - rng.randrange(8, max(9, min(45, width // 20)))
        bottom = height - rng.randrange(8, max(9, min(45, height // 20)))
        return image.crop((left, top, right, bottom)).resize(
            (width, height), Image.Resampling.LANCZOS
        )
    if augmentation_type == "slight_rotation":
        angle = rng.choice((-1, 1)) * rng.uniform(1.0, 2.0)
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
    raise ValueError(f"Unknown augmentation: {augmentation_type}")


def prepare_augmentation(
    parent: PreparedDocument,
    document_id: str,
    augmentation_type: str,
    staging_root: Path,
    args: argparse.Namespace,
) -> PreparedDocument:
    raw_stage, image_stage, text_stage, ocr_stage = staged_paths(
        staging_root, parent.label, document_id, ".png"
    )
    raw_final, _, _, _ = final_paths(parent.label, document_id, ".png")
    with Image.open(parent.image_stage_path) as source_image:
        augmented = apply_augmentation(
            source_image,
            augmentation_type,
            stable_rng(args.seed, f"{parent.document_id}:{augmentation_type}:{document_id}"),
        )
    raw_stage.parent.mkdir(parents=True, exist_ok=True)
    augmented.save(raw_stage, format="PNG", optimize=True)
    if augmentation_type in {"screenshot_crop", "slight_rotation"}:
        jobs_dir = staging_root / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job_path = jobs_dir / f"{document_id}.json"
        job_path.write_text(
            json.dumps(
                {
                    "raw_path": str(raw_stage),
                    "label": parent.label,
                    "image_path": str(image_stage),
                    "text_path": str(text_stage),
                    "ocr_path": str(ocr_stage),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        success, reason, _ = run_preprocess_worker(job_path, args.per_document_timeout)
        if not success:
            raise DatasetExpansionError(reason)
    else:
        image_stage.parent.mkdir(parents=True, exist_ok=True)
        text_stage.parent.mkdir(parents=True, exist_ok=True)
        ocr_stage.parent.mkdir(parents=True, exist_ok=True)
        augmented.save(image_stage, format="PNG", optimize=True)
        shutil.copy2(parent.text_stage_path, text_stage)
        payload = json.loads(parent.ocr_stage_path.read_text(encoding="utf-8"))
        payload["label"] = parent.label
        payload["image_width"] = augmented.width
        payload["image_height"] = augmented.height
        ocr_stage.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    valid, reason = processed_outputs_valid(image_stage, text_stage, ocr_stage)
    if not valid:
        raise DatasetExpansionError(reason)
    return PreparedDocument(
        document_id=document_id,
        label=parent.label,
        source_name=parent.source_name,
        source_locator=parent.source_locator,
        original_id=parent.original_id,
        language="hr",
        is_synthetic=parent.is_synthetic,
        is_augmented=True,
        augmentation_type=augmentation_type,
        parent_id=parent.document_id,
        raw_stage_path=raw_stage,
        image_stage_path=image_stage,
        text_stage_path=text_stage,
        ocr_stage_path=ocr_stage,
        raw_final_path=raw_final,
    )


def fingerprint_prepared(document: PreparedDocument):
    record = build_fingerprint_record(
        key=document.document_id,
        raw_path=document.raw_stage_path,
        image_path=document.image_stage_path,
        text_path=document.text_stage_path,
        label=document.label,
        source=document.source_name,
        group_id=document.parent_id or document.document_id,
    )
    record.path = str(document.raw_final_path)
    return record


def metadata_row(document: PreparedDocument) -> dict[str, str]:
    _, image_final, text_final, ocr_final = final_paths(
        document.label, document.document_id, document.raw_final_path.suffix
    )
    return {
        "id": document.document_id,
        "label": document.label,
        "raw_path": relative_project_path(PROJECT_ROOT, document.raw_final_path),
        "image_path": relative_project_path(PROJECT_ROOT, image_final),
        "text_path": relative_project_path(PROJECT_ROOT, text_final),
        "ocr_path": relative_project_path(PROJECT_ROOT, ocr_final),
    }


def source_row(document: PreparedDocument) -> dict[str, str]:
    return {
        "id": document.document_id,
        "label": document.label,
        "raw_path": relative_project_path(PROJECT_ROOT, document.raw_final_path),
        "source_name": document.source_name,
        "source_url_or_dataset": document.source_locator,
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "original_id": document.parent_id if document.is_augmented else document.original_id,
        "language": document.language,
        "is_synthetic": str(document.is_synthetic),
        "is_augmented": str(document.is_augmented),
        "augmentation_type": document.augmentation_type,
        "duplicate_check_status": "passed",
    }


def normalize_source_row(row: Mapping[str, Any]) -> dict[str, str]:
    normalized = {field: str(row.get(field, "") or "") for field in SOURCE_FIELDS}
    if not normalized["language"] and normalized["source_name"] in HR_SOURCE_NAMES:
        normalized["language"] = "hr"
    if not normalized["is_synthetic"]:
        normalized["is_synthetic"] = str(
            normalized["source_name"].startswith("hr_synthetic_")
        )
    return normalized


def read_source_rows() -> list[dict[str, str]]:
    if not SOURCE_TRACKING_PATH.exists():
        return []
    with SOURCE_TRACKING_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return [normalize_source_row(row) for row in csv.DictReader(handle)]


def is_hr_source(row: Mapping[str, str]) -> bool:
    return (
        row.get("language", "").casefold() in {"hr", "hrv", "cro"}
        or row.get("source_name", "") in HR_SOURCE_NAMES
        or "_hr_" in row.get("id", "")
    )


def augmentation_parent_map(rows: Iterable[Mapping[str, str]]) -> dict[str, str]:
    return {
        str(row.get("id", "")): str(row.get("original_id", ""))
        for row in rows
        if str(row.get("is_augmented", "")).casefold() in {"true", "1", "yes"}
        and row.get("id")
        and row.get("original_id")
    }


def commit_documents(
    documents: Sequence[PreparedDocument],
    *,
    skip_existing: bool,
) -> None:
    copy_pairs: list[tuple[Path, Path]] = []
    for document in documents:
        _, image_final, text_final, ocr_final = final_paths(
            document.label, document.document_id, document.raw_final_path.suffix
        )
        pairs = (
            (document.raw_stage_path, document.raw_final_path),
            (document.image_stage_path, image_final),
            (document.text_stage_path, text_final),
            (document.ocr_stage_path, ocr_final),
        )
        for source, destination in pairs:
            validate_training_path(PROJECT_ROOT, destination)
            if not source.exists():
                raise FileNotFoundError(f"Missing staged output: {source}")
            if destination.exists():
                if not skip_existing:
                    raise FileExistsError(f"Refusing to overwrite: {destination}")
                if sha256_file(source) != sha256_file(destination):
                    raise FileExistsError(f"Existing file differs from staged output: {destination}")
                continue
            copy_pairs.append((source, destination))
    for source, destination in copy_pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def verify_split_integrity(
    metadata_rows: Sequence[Mapping[str, str]],
    splits: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    expected_ids = {str(row["id"]) for row in metadata_rows}
    split_sets = {
        name: {str(row["id"]) for row in rows}
        for name, rows in splits.items()
    }
    if set().union(*split_sets.values()) != expected_ids:
        raise DatasetExpansionError("Generated splits do not contain exactly all metadata IDs")
    names = list(split_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise DatasetExpansionError(
                    f"Generated splits overlap ({left}/{right}): {sorted(overlap)[:5]}"
                )


def commit_class_batch(
    documents: Sequence[PreparedDocument],
    metadata_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    additions = [metadata_row(document) for document in documents]
    source_additions = [source_row(document) for document in documents]
    known_ids = {row["id"] for row in metadata_rows}
    collisions = [row["id"] for row in additions if row["id"] in known_ids]
    if collisions:
        raise DatasetExpansionError(f"Metadata ID collision: {collisions[:5]}")
    all_metadata = metadata_rows + additions
    all_sources = source_rows + source_additions
    splits = group_aware_stratified_split(
        all_metadata,
        augmentation_parent_map(all_sources),
        seed=args.seed,
    )
    verify_split_integrity(all_metadata, splits)
    commit_documents(documents, skip_existing=args.skip_existing)
    atomic_write_csv(METADATA_PATH, all_metadata, METADATA_FIELDS)
    atomic_write_csv(SOURCE_TRACKING_PATH, all_sources, SOURCE_FIELDS)
    for split_name, split_rows in splits.items():
        atomic_write_csv(SPLITS_DIR / f"{split_name}.csv", split_rows, METADATA_FIELDS)
    return all_metadata, all_sources, splits


def read_duplicate_rows() -> list[dict[str, object]]:
    if not HR_DUPLICATES_PATH.exists():
        return []
    with HR_DUPLICATES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {field: str(row.get(field, "") or "") for field in HR_DUPLICATE_FIELDS}
            for row in csv.DictReader(handle)
        ]


def deduplicated_report_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        normalized = {field: str(row.get(field, "") or "") for field in HR_DUPLICATE_FIELDS}
        key = tuple(normalized[field] for field in HR_DUPLICATE_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def snapshot_external_data() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for folder_name in ("external_test", "external_robus_test"):
        root = DATA_DIR / folder_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                snapshot[relative_project_path(PROJECT_ROOT, path)] = sha256_file(path)
    return snapshot


def ensure_holdout_fingerprints(
    index: FingerprintIndex,
    report_rows: list[dict[str, object]],
) -> None:
    indexed_paths = {Path(record.path).resolve() for record in index.records}
    for folder_name in ("external_test", "external_robus_test"):
        root = DATA_DIR / folder_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in {
                ".pdf", ".png", ".jpg", ".jpeg", ".txt", ".html", ".htm", ".docx"
            }:
                continue
            if path.resolve() in indexed_paths:
                continue
            label = next((part for part in path.parts if part in CLASS_NAMES), "")
            try:
                record = build_fingerprint_record(
                    key=f"holdout:{relative_project_path(PROJECT_ROOT, path)}",
                    raw_path=path,
                    label=label,
                    source=f"{folder_name}_guard",
                )
                index.add(record)
                indexed_paths.add(path.resolve())
            except Exception as error:
                add_duplicate_report(
                    report_rows,
                    candidate_id="",
                    candidate_path=relative_project_path(PROJECT_ROOT, path),
                    label=label,
                    source_name=f"{folder_name}_guard",
                    decision="warning",
                    reason=f"holdout_fingerprint_error: {error}",
                )


def initialize_id_counters(existing_ids: Iterable[str]) -> dict[tuple[str, bool], int]:
    counters: dict[tuple[str, bool], int] = {}
    ids = list(existing_ids)
    for label, prefix in ID_PREFIXES.items():
        for augmented in (False, True):
            marker = f"{prefix}{'_aug' if augmented else ''}_"
            numbers = [
                int(match.group(1))
                for document_id in ids
                if (match := re.fullmatch(re.escape(marker) + r"(\d+)", document_id))
            ]
            counters[(label, augmented)] = max(numbers, default=0) + 1
    return counters


def allocate_document_id(
    label: str,
    augmented: bool,
    counters: dict[tuple[str, bool], int],
    reserved_ids: set[str],
) -> str:
    prefix = ID_PREFIXES[label] + ("_aug" if augmented else "")
    while True:
        number = counters[(label, augmented)]
        counters[(label, augmented)] += 1
        document_id = f"{prefix}_{number:04d}"
        if document_id not in reserved_ids:
            reserved_ids.add(document_id)
            return document_id


def prepared_from_existing(
    metadata: Mapping[str, str], source: Mapping[str, str]
) -> PreparedDocument | None:
    try:
        raw_path = project_path(PROJECT_ROOT, metadata["raw_path"])
        image_path = project_path(PROJECT_ROOT, metadata["image_path"])
        text_path = project_path(PROJECT_ROOT, metadata["text_path"])
        ocr_path = project_path(PROJECT_ROOT, metadata["ocr_path"])
        valid, _ = processed_outputs_valid(image_path, text_path, ocr_path)
        if not raw_path.exists() or not valid:
            return None
        return PreparedDocument(
            document_id=str(metadata["id"]),
            label=str(metadata["label"]),
            source_name=str(source.get("source_name", "")),
            source_locator=str(source.get("source_url_or_dataset", "")),
            original_id=str(source.get("original_id", metadata["id"])),
            language=str(source.get("language", "hr") or "hr"),
            is_synthetic=str(source.get("is_synthetic", "")).casefold() in {"true", "1", "yes"},
            is_augmented=False,
            augmentation_type="",
            parent_id="",
            raw_stage_path=raw_path,
            image_stage_path=image_path,
            text_stage_path=text_path,
            ocr_stage_path=ocr_path,
            raw_final_path=raw_path,
        )
    except Exception:
        return None


def write_hr_reports(
    *,
    metadata_rows: Sequence[Mapping[str, str]],
    source_rows: Sequence[Mapping[str, str]],
    duplicate_rows: Sequence[Mapping[str, object]],
    added_this_run: set[str],
    target: int,
    external_unchanged: bool,
) -> None:
    metadata_ids = {str(row["id"]) for row in metadata_rows}
    hr_sources = [
        normalize_source_row(row)
        for row in source_rows
        if row.get("id") in metadata_ids and is_hr_source(row)
    ]
    duplicate_rows = deduplicated_report_rows(duplicate_rows)
    totals = class_counts(metadata_rows)
    lines = [
        "HR DATASET EXPANSION SUMMARY",
        "=" * 80,
        f"Target Croatian documents per class: {target}",
        f"External test data unchanged: {'YES' if external_unchanged else 'NO'}",
    ]
    for label in CLASS_NAMES:
        rows = [row for row in hr_sources if row["label"] == label]
        real_count = sum(row["is_synthetic"].casefold() not in {"true", "1", "yes"} for row in rows)
        synthetic_count = len(rows) - real_count
        augmented_count = sum(row["is_augmented"].casefold() in {"true", "1", "yes"} for row in rows)
        duplicate_count = sum(
            row.get("label") == label
            and row.get("decision") == "skipped"
            and str(row.get("reason", ""))
            in {
                "identical_sha256",
                "identical_normalized_text",
                "near_identical_first_page",
                "near_duplicate_text",
            }
            for row in duplicate_rows
        )
        source_names = sorted({row["source_name"] for row in rows if row["source_name"]})
        added_count = sum(row["id"] in added_this_run for row in rows)
        lines.extend(
            [
                "",
                label,
                f"  Croatian documents tracked: {len(rows)}",
                f"  Added in this run: {added_count}",
                f"  Public/real: {real_count}",
                f"  Synthetic: {synthetic_count}",
                f"  Augmented: {augmented_count}",
                f"  Skipped duplicate/similar: {duplicate_count}",
                f"  Total documents in class: {totals[label]}",
                f"  Sources: {', '.join(source_names) or 'none'}",
                f"  Status: {'TARGET REACHED' if len(rows) >= target else 'BELOW TARGET'}",
            ]
        )
    lines.extend(
        [
            "",
            f"Total Croatian documents tracked: {len(hr_sources)}",
            f"Duplicate/report entries: {len(duplicate_rows)}",
            f"Source tracking: {relative_project_path(PROJECT_ROOT, SOURCE_TRACKING_PATH)}",
            f"HR source report: {relative_project_path(PROJECT_ROOT, HR_SOURCES_PATH)}",
            f"HR duplicate report: {relative_project_path(PROJECT_ROOT, HR_DUPLICATES_PATH)}",
        ]
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(HR_SOURCES_PATH, hr_sources, SOURCE_FIELDS)
    atomic_write_csv(HR_DUPLICATES_PATH, duplicate_rows, HR_DUPLICATE_FIELDS)
    atomic_write_text(HR_SUMMARY_PATH, "\n".join(lines) + "\n")


def dry_run_report(
    args: argparse.Namespace,
    metadata_rows: Sequence[Mapping[str, str]],
    source_rows: Sequence[Mapping[str, str]],
) -> None:
    metadata_ids = {str(row["id"]) for row in metadata_rows}
    hr_rows = [row for row in source_rows if row.get("id") in metadata_ids and is_hr_source(row)]
    print("DRY RUN - no files will be downloaded, generated, or changed")
    print(f"Metadata rows: {len(metadata_rows)}")
    print(f"Requested Croatian target per class: {args.target_per_class}")
    for label in CLASS_NAMES:
        existing = sum(row.get("label") == label for row in hr_rows)
        remaining = max(0, args.target_per_class - existing)
        desired_augmentations = min(
            int(args.target_per_class * args.augmentation_fraction),
            int(args.target_per_class * 0.20),
        )
        existing_augmentations = sum(
            row.get("label") == label
            and row.get("is_augmented", "").casefold() in {"true", "1", "yes"}
            for row in hr_rows
        )
        planned_augmentations = min(remaining, max(0, desired_augmentations - existing_augmentations))
        planned_base = remaining - planned_augmentations
        strategy = {
            "invoice": "synthetic Croatian invoices",
            "cv": "synthetic Croatian CVs",
            "contract": "data.gov.hr procurement records + synthetic fallback",
            "email": "synthetic Croatian webmail screenshots",
            "scientific": "Hrčak Croatian PDFs + synthetic fallback",
        }[label]
        print(
            f"{label:12s} existing_hr={existing:3d} add_base={planned_base:3d} "
            f"add_augmented={planned_augmentations:3d} source={strategy}"
        )
    print("Protected folders: data/external_test, data/external_robus_test")


def collect_public_candidates_for_label(
    label: str,
    needed: int,
    client: HttpClient,
    args: argparse.Namespace,
    report_rows: list[dict[str, object]],
) -> list[Candidate]:
    try:
        if label == "scientific":
            if not args.hrcak_download_approved:
                add_duplicate_report(
                    report_rows,
                    candidate_id="",
                    label="scientific",
                    source_name="hrcak",
                    decision="warning",
                    reason="hrcak_content_mining_permission_not_confirmed",
                )
                print(
                    "UPOZORENJE: Hrčak PDF dohvat nije uključen jer nije potvrđeno "
                    "ispunjavanje uvjeta portala; koristim sintetički fallback."
                )
                return []
            return collect_hrcak_candidates(client, needed, args.max_hrcak_pages, report_rows)
        if label == "contract":
            return collect_contract_candidates(client, needed, report_rows)
    except Exception as error:
        add_duplicate_report(
            report_rows,
            candidate_id="",
            label=label,
            source_name="hrcak" if label == "scientific" else "hrvatska_javna_nabava",
            decision="warning",
            reason=f"public_source_error: {error}",
        )
        print(f"UPOZORENJE: javni izvor za {label} nije dostupan: {error}")
    return []


def accept_base_documents(
    *,
    label: str,
    count: int,
    public_candidates: deque[Candidate],
    synthetic_stream: Iterator[Candidate],
    staging_root: Path,
    client: HttpClient,
    args: argparse.Namespace,
    index: FingerprintIndex,
    report_rows: list[dict[str, object]],
    counters: dict[tuple[str, bool], int],
    reserved_ids: set[str],
) -> list[PreparedDocument]:
    accepted: list[PreparedDocument] = []
    attempts = 0
    max_attempts = max(300, count * 25 + len(public_candidates))
    progress = tqdm(total=count, desc=f"Dodavanje HR {label}", unit="doc")
    permanent_source_failures: Counter[str] = Counter()
    try:
        while len(accepted) < count and attempts < max_attempts:
            attempts += 1
            candidate = public_candidates.popleft() if public_candidates else next(synthetic_stream)
            document_id = allocate_document_id(label, False, counters, reserved_ids)
            candidate_path = relative_project_path(
                PROJECT_ROOT,
                RAW_DIR / label / f"{document_id}{candidate.extension}",
            )
            try:
                document = prepare_candidate(candidate, document_id, staging_root, client, args)
                record = fingerprint_prepared(document)
                match = index.find_duplicate(record)
                if match:
                    add_duplicate_report(
                        report_rows,
                        candidate_id=document_id,
                        candidate_path=candidate_path,
                        label=label,
                        source_name=candidate.source_name,
                        decision="skipped",
                        reason=match.reason,
                        similar_to=match.similar_to,
                        similarity_score=match.similarity_score,
                    )
                    continue
                index.add(record)
                accepted.append(document)
                progress.update(1)
                if len(accepted) % 10 == 0:
                    progress.set_postfix(source=candidate.source_name)
            except Exception as error:
                add_duplicate_report(
                    report_rows,
                    candidate_id=document_id,
                    candidate_path=candidate_path,
                    label=label,
                    source_name=candidate.source_name,
                    decision="skipped",
                    reason=f"preparation_error: {error}",
                )
                if candidate.source_name in {"hrcak", "hrvatska_javna_nabava"}:
                    tqdm.write(f"SKIPPED {candidate.source_name}: {candidate.source_locator} ({error})")
                    if isinstance(error, RemoteAccessDenied) or "access denied" in str(error).casefold():
                        permanent_source_failures[candidate.source_name] += 1
                        if permanent_source_failures[candidate.source_name] >= 1:
                            public_candidates = deque(
                                item
                                for item in public_candidates
                                if item.source_name != candidate.source_name
                            )
                            tqdm.write(
                                f"Javni izvor {candidate.source_name} trajno odbija pristup; "
                                "nastavljam sa sigurnim fallbackom."
                            )
    finally:
        progress.close()
    if len(accepted) != count:
        raise DatasetExpansionError(
            f"Could prepare only {len(accepted)}/{count} base Croatian documents for {label}"
        )
    return accepted


def accept_augmentations(
    *,
    label: str,
    count: int,
    parents: Sequence[PreparedDocument],
    staging_root: Path,
    args: argparse.Namespace,
    index: FingerprintIndex,
    report_rows: list[dict[str, object]],
    counters: dict[tuple[str, bool], int],
    reserved_ids: set[str],
) -> list[PreparedDocument]:
    if count <= 0:
        return []
    usable_parents = list(parents)
    stable_rng(args.seed, f"parents:{label}").shuffle(usable_parents)
    if not usable_parents:
        return []
    accepted: list[PreparedDocument] = []
    attempts = 0
    max_attempts = max(100, count * 10)
    progress = tqdm(total=count, desc=f"HR augmentacije {label}", unit="doc")
    try:
        while len(accepted) < count and attempts < max_attempts:
            parent = usable_parents[attempts % len(usable_parents)]
            augmentation_type = AUGMENTATION_TYPES[attempts % len(AUGMENTATION_TYPES)]
            attempts += 1
            document_id = allocate_document_id(label, True, counters, reserved_ids)
            candidate_path = relative_project_path(
                PROJECT_ROOT, RAW_DIR / label / f"{document_id}.png"
            )
            try:
                document = prepare_augmentation(
                    parent,
                    document_id,
                    augmentation_type,
                    staging_root,
                    args,
                )
                record = fingerprint_prepared(document)
                match = index.find_duplicate(
                    record,
                    ignore={parent.document_id, str(parent.raw_final_path)},
                )
                if match:
                    add_duplicate_report(
                        report_rows,
                        candidate_id=document_id,
                        candidate_path=candidate_path,
                        label=label,
                        source_name=parent.source_name,
                        decision="skipped",
                        reason=match.reason,
                        similar_to=match.similar_to,
                        similarity_score=match.similarity_score,
                    )
                    continue
                index.add(record)
                accepted.append(document)
                progress.update(1)
            except Exception as error:
                add_duplicate_report(
                    report_rows,
                    candidate_id=document_id,
                    candidate_path=candidate_path,
                    label=label,
                    source_name=parent.source_name,
                    decision="skipped",
                    reason=f"augmentation_error: {error}",
                    similar_to=parent.document_id,
                )
    finally:
        progress.close()
    return accepted


def main() -> None:
    args = parse_args()
    if args.worker_job:
        run_worker_job(args.worker_job)
        return

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA_PATH}")
    metadata_rows = read_csv_rows(METADATA_PATH, METADATA_FIELDS)
    source_rows = read_source_rows()
    metadata_ids = {row["id"] for row in metadata_rows}
    if len(metadata_ids) != len(metadata_rows):
        raise DatasetExpansionError("data/metadata.csv contains duplicate IDs")
    hr_rows = [row for row in source_rows if row.get("id") in metadata_ids and is_hr_source(row)]
    if args.dry_run:
        dry_run_report(args, metadata_rows, source_rows)
        return
    if hr_rows and not args.skip_existing:
        raise DatasetExpansionError(
            "Croatian additions already exist. Re-run with --skip-existing to resume safely."
        )
    if args.augmentation_fraction > 0 and not TESSERACT_AVAILABLE:
        raise DatasetExpansionError(
            "Tesseract is required for augmented OCR outputs but is not available."
        )

    before_external = snapshot_external_data()
    report_rows = read_duplicate_rows()
    print(f"Učitavam fingerprint indeks za {len(metadata_rows)} postojećih dokumenata...")
    index, fingerprint_failures = load_existing_fingerprint_index(PROJECT_ROOT, metadata_rows)
    for failure in fingerprint_failures:
        add_duplicate_report(
            report_rows,
            candidate_id=str(failure.get("id", "")),
            candidate_path=str(failure.get("raw_path", "")),
            label="",
            source_name="existing_dataset",
            decision="warning",
            reason=f"fingerprint_error: {failure.get('error', '')}",
        )
    ensure_holdout_fingerprints(index, report_rows)
    print(f"Fingerprint indeks: {len(index.records)} dokumenata uključujući holdout zaštitu.")

    reserved_ids = set(metadata_ids)
    counters = initialize_id_counters(reserved_ids)
    added_this_run: set[str] = set()
    client = HttpClient(args.request_timeout)
    try:
        for label in CLASS_NAMES:
            source_by_id = {row["id"]: row for row in source_rows}
            existing_hr = [
                row for row in source_rows
                if row.get("label") == label
                and row.get("id") in {item["id"] for item in metadata_rows}
                and is_hr_source(row)
            ]
            remaining = max(0, args.target_per_class - len(existing_hr))
            print(
                f"\n{label.upper()}: HR postojeći={len(existing_hr)}, potrebno dodati={remaining}"
            )
            if remaining == 0:
                continue

            desired_augmentation_total = min(
                int(args.target_per_class * args.augmentation_fraction),
                int(args.target_per_class * 0.20),
            )
            existing_augmentations = sum(
                row.get("is_augmented", "").casefold() in {"true", "1", "yes"}
                for row in existing_hr
            )
            augmentation_needed = min(
                remaining,
                max(0, desired_augmentation_total - existing_augmentations),
            )
            base_needed = remaining - augmentation_needed

            used_original_ids = {
                row.get("original_id", "")
                for row in existing_hr
                if row.get("is_augmented", "").casefold() not in {"true", "1", "yes"}
            }
            public_candidates = deque(
                candidate
                for candidate in collect_public_candidates_for_label(
                    label,
                    base_needed,
                    client,
                    args,
                    report_rows,
                )
                if candidate.original_id not in used_original_ids
            )
            synthetic_stream = synthetic_candidates(
                label,
                args.seed,
                start=next_synthetic_original_index(label, args.seed, existing_hr),
            )

            with tempfile.TemporaryDirectory(prefix=f"document_ai_hr_{label}_") as temporary:
                staging_root = Path(temporary)
                base_documents = accept_base_documents(
                    label=label,
                    count=base_needed,
                    public_candidates=public_candidates,
                    synthetic_stream=synthetic_stream,
                    staging_root=staging_root,
                    client=client,
                    args=args,
                    index=index,
                    report_rows=report_rows,
                    counters=counters,
                    reserved_ids=reserved_ids,
                )

                existing_parent_documents: list[PreparedDocument] = []
                metadata_by_id = {row["id"]: row for row in metadata_rows}
                for row in existing_hr:
                    if row.get("is_augmented", "").casefold() in {"true", "1", "yes"}:
                        continue
                    metadata = metadata_by_id.get(row["id"])
                    if metadata:
                        parent = prepared_from_existing(metadata, source_by_id[row["id"]])
                        if parent:
                            existing_parent_documents.append(parent)
                augmented_documents = accept_augmentations(
                    label=label,
                    count=augmentation_needed,
                    parents=existing_parent_documents + base_documents,
                    staging_root=staging_root,
                    args=args,
                    index=index,
                    report_rows=report_rows,
                    counters=counters,
                    reserved_ids=reserved_ids,
                )

                augmentation_shortfall = augmentation_needed - len(augmented_documents)
                if augmentation_shortfall:
                    print(
                        f"UPOZORENJE: {label} ima {augmentation_shortfall} neuspjelih "
                        "augmentacija; cilj popunjavam dodatnim izvornim dokumentima."
                    )
                    base_documents.extend(
                        accept_base_documents(
                            label=label,
                            count=augmentation_shortfall,
                            public_candidates=public_candidates,
                            synthetic_stream=synthetic_stream,
                            staging_root=staging_root,
                            client=client,
                            args=args,
                            index=index,
                            report_rows=report_rows,
                            counters=counters,
                            reserved_ids=reserved_ids,
                        )
                    )

                documents = base_documents + augmented_documents
                if len(documents) != remaining:
                    raise DatasetExpansionError(
                        f"Prepared {len(documents)}/{remaining} required documents for {label}"
                    )
                metadata_rows, source_rows, _ = commit_class_batch(
                    documents,
                    metadata_rows,
                    source_rows,
                    args,
                )
                added_this_run.update(document.document_id for document in documents)

            current_external = snapshot_external_data()
            if current_external != before_external:
                raise DatasetExpansionError(
                    "Protected external test data changed during the run; stopping immediately."
                )
            write_hr_reports(
                metadata_rows=metadata_rows,
                source_rows=source_rows,
                duplicate_rows=report_rows,
                added_this_run=added_this_run,
                target=args.target_per_class,
                external_unchanged=True,
            )
            print(f"{label}: commitano {remaining} dokumenata u postojeće raw/processed foldere.")
    finally:
        client.close()

    after_external = snapshot_external_data()
    external_unchanged = before_external == after_external
    write_hr_reports(
        metadata_rows=metadata_rows,
        source_rows=source_rows,
        duplicate_rows=report_rows,
        added_this_run=added_this_run,
        target=args.target_per_class,
        external_unchanged=external_unchanged,
    )
    if not external_unchanged:
        raise DatasetExpansionError("External test snapshot verification failed")

    final_hr_rows = [
        row for row in source_rows
        if row.get("id") in {item["id"] for item in metadata_rows} and is_hr_source(row)
    ]
    print("\nHR DATASET JE PRIPREMLJEN")
    final_counts = class_counts(metadata_rows)
    for label in CLASS_NAMES:
        hr_count = sum(row.get("label") == label for row in final_hr_rows)
        print(f"{label:12s} HR={hr_count:3d}  ukupno={final_counts[label]:4d}")
    print(f"Metadata: {relative_project_path(PROJECT_ROOT, METADATA_PATH)}")
    print(f"Splitovi: {relative_project_path(PROJECT_ROOT, SPLITS_DIR)}")
    print(f"Sažetak: {relative_project_path(PROJECT_ROOT, HR_SUMMARY_PATH)}")
    print(f"Izvori: {relative_project_path(PROJECT_ROOT, HR_SOURCES_PATH)}")
    print(f"Duplikati: {relative_project_path(PROJECT_ROOT, HR_DUPLICATES_PATH)}")
    print("Modeli nisu trenirani.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrekinuto od korisnika; već commitane klase ostaju sačuvane.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"GREŠKA: {error}", file=sys.stderr)
        raise SystemExit(1)
