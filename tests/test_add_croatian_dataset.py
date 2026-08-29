import io
import random
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.add_croatian_dataset import (
    Candidate,
    apply_augmentation,
    fake_oib,
    normalize_source_row,
    next_synthetic_original_index,
    parse_args,
    render_candidate,
    rows_from_xlsx,
    sanitize_public_value,
)
from src.dataset_expansion import FingerprintIndex, build_fingerprint_record


def oib_is_valid(value: str) -> bool:
    if len(value) != 11 or not value.isdigit():
        return False
    remainder = 10
    for character in value[:10]:
        remainder = (remainder + int(character)) % 10
        if remainder == 0:
            remainder = 10
        remainder = (remainder * 2) % 11
    check = 11 - remainder
    if check == 10:
        check = 0
    return check == int(value[-1])


class CroatianDatasetTests(unittest.TestCase):
    def test_generated_oibs_are_intentionally_invalid(self):
        rng = random.Random(42)
        values = [fake_oib(rng) for _ in range(100)]
        self.assertTrue(all(len(value) == 11 and value.isdigit() for value in values))
        self.assertTrue(all(not oib_is_valid(value) for value in values))

    def test_public_procurement_values_remove_contact_identifiers(self):
        raw = (
            "Kontakt ana.primjer@example.com, +385 91 123 4567, "
            "OIB 12345678901, IBAN HR12 1234567 12345 1234567"
        )
        sanitized = sanitize_public_value(raw)
        self.assertNotIn("@", sanitized)
        self.assertNotIn("12345678901", sanitized)
        self.assertNotIn("HR12", sanitized)
        self.assertIn("uklonjen", sanitized)

    def test_synthetic_renderers_produce_aligned_words_and_boxes(self):
        candidates = (
            Candidate("invoice", "hr_synthetic_invoice", "generated://invoice", "invoice-a", "invoice", True),
            Candidate("cv", "hr_synthetic_cv", "generated://cv", "cv-a", "cv", True),
            Candidate("email", "hr_synthetic_gmail_like", "generated://email", "email-a", "email", True, extension=".png"),
            Candidate("contract", "hr_synthetic_contract", "generated://contract", "contract-a", "synthetic_contract", True),
            Candidate("scientific", "hr_synthetic_scientific", "generated://scientific", "scientific-a", "synthetic_scientific", True),
        )
        for candidate in candidates:
            with self.subTest(label=candidate.label):
                rendered = render_candidate(candidate, seed=42)
                self.assertGreater(len(rendered.words), 20)
                self.assertEqual(len(rendered.words), len(rendered.boxes))
                self.assertGreater(len(rendered.text), 100)
                self.assertEqual(rendered.image.mode, "RGB")

    def test_invoice_variants_are_not_pixel_identical(self):
        first = render_candidate(
            Candidate("invoice", "hr_synthetic_invoice", "generated://invoice", "invoice-one", "invoice", True),
            seed=42,
        )
        second = render_candidate(
            Candidate("invoice", "hr_synthetic_invoice", "generated://invoice", "invoice-two", "invoice", True),
            seed=42,
        )
        self.assertNotEqual(first.image.tobytes(), second.image.tobytes())

    def test_scientific_generator_produces_a_diverse_candidate_pool(self):
        index = FingerprintIndex()
        accepted = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in range(1, 51):
                rendered = render_candidate(
                    Candidate(
                        "scientific",
                        "hr_synthetic_scientific",
                        "generated://scientific",
                        f"generated:scientific:hr:42:{number:06d}",
                        "synthetic_scientific",
                        True,
                    ),
                    seed=42,
                )
                image_path = root / f"scientific_{number:04d}.png"
                text_path = root / f"scientific_{number:04d}.txt"
                rendered.image.save(image_path)
                text_path.write_text(rendered.text, encoding="utf-8")
                record = build_fingerprint_record(
                    key=str(number),
                    raw_path=image_path,
                    image_path=image_path,
                    text_path=text_path,
                    label="scientific",
                    source="fixture",
                )
                if index.find_duplicate(record) is None:
                    index.add(record)
                    accepted += 1
        self.assertGreaterEqual(accepted, 48)

    def test_other_synthetic_generators_produce_diverse_candidate_pools(self):
        fixtures = {
            "invoice": ("hr_synthetic_invoice", "invoice"),
            "cv": ("hr_synthetic_cv", "cv"),
            "email": ("hr_synthetic_gmail_like", "email"),
            "contract": ("hr_synthetic_contract", "synthetic_contract"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, (source, kind) in fixtures.items():
                index = FingerprintIndex()
                accepted = 0
                reasons = Counter()
                for number in range(1, 41):
                    rendered = render_candidate(
                        Candidate(
                            label,
                            source,
                            f"generated://{label}",
                            f"generated:{label}:hr:42:{number:06d}",
                            kind,
                            True,
                        ),
                        seed=42,
                    )
                    image_path = root / f"{label}_{number:04d}.png"
                    text_path = root / f"{label}_{number:04d}.txt"
                    rendered.image.save(image_path)
                    text_path.write_text(rendered.text, encoding="utf-8")
                    record = build_fingerprint_record(
                        key=f"{label}-{number}",
                        raw_path=image_path,
                        image_path=image_path,
                        text_path=text_path,
                        label=label,
                        source="fixture",
                    )
                    match = index.find_duplicate(record)
                    if match is None:
                        index.add(record)
                        accepted += 1
                    else:
                        reasons[match.reason] += 1
                with self.subTest(label=label, reasons=dict(reasons)):
                    self.assertGreaterEqual(
                        accepted,
                        36,
                        f"{label}: accepted={accepted}, reasons={dict(reasons)}",
                    )

    def test_minimal_xlsx_is_read_without_openpyxl(self):
        shared_strings = """<?xml version="1.0" encoding="UTF-8"?>
        <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <si><t>Predmet ugovora</t></si><si><t>Vrijednost</t></si>
          <si><t>Nabava opreme</t></si><si><t>12500 EUR</t></si>
        </sst>"""
        worksheet = """<?xml version="1.0" encoding="UTF-8"?>
        <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <sheetData>
            <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
            <row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" t="s"><v>3</v></c></row>
          </sheetData>
        </worksheet>"""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("xl/sharedStrings.xml", shared_strings)
            archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        rows = rows_from_xlsx(buffer.getvalue())
        self.assertEqual(rows, [{"Predmet ugovora": "Nabava opreme", "Vrijednost": "12500 EUR"}])

    def test_legacy_source_rows_receive_new_fields(self):
        row = normalize_source_row(
            {
                "id": "invoice_extra_0001",
                "label": "invoice",
                "source_name": "rvl_cdip_invoice",
                "is_augmented": "False",
            }
        )
        self.assertIn("language", row)
        self.assertEqual(row["is_synthetic"], "False")

    def test_resume_starts_after_highest_synthetic_original_id(self):
        rows = [
            {"original_id": "generated:cv:hr:42:000004", "is_augmented": "False"},
            {"original_id": "generated:cv:hr:42:000011", "is_augmented": "False"},
            {"original_id": "cv_hr_synthetic_0004", "is_augmented": "True"},
        ]
        self.assertEqual(next_synthetic_original_index("cv", 42, rows), 12)

    def test_augmentation_keeps_image_dimensions(self):
        from PIL import Image

        image = Image.new("RGB", (400, 600), "white")
        for name in (
            "brightness_low",
            "slight_blur",
            "low_contrast",
            "jpeg_compression",
            "screenshot_crop",
            "slight_rotation",
        ):
            with self.subTest(name=name):
                output = apply_augmentation(image, name, random.Random(42))
                self.assertEqual(output.size, image.size)

    def test_augmentation_fraction_cannot_exceed_twenty_percent(self):
        with patch.object(
            sys,
            "argv",
            ["add_croatian_dataset.py", "--augmentation-fraction", "0.21"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
