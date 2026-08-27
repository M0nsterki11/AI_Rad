import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from PIL import Image

from src import predict_text_model, preprocess


class TesseractRuntimeTests(unittest.TestCase):
    def test_availability_check_calls_tesseract(self):
        with patch.object(
            preprocess.pytesseract,
            "get_tesseract_version",
            return_value="5.0.0",
        ) as get_version:
            self.assertTrue(preprocess.is_tesseract_available())
            get_version.assert_called_once_with()

        with patch.object(
            preprocess.pytesseract,
            "get_tesseract_version",
            side_effect=OSError("missing executable"),
        ):
            self.assertFalse(preprocess.is_tesseract_available())

    def test_unavailable_tesseract_raises_clear_error(self):
        with patch.object(preprocess, "TESSERACT_AVAILABLE", False):
            with self.assertRaisesRegex(
                preprocess.OCRProcessingError,
                "Tesseract OCR nije dostupan",
            ):
                preprocess.run_ocr_on_image(Image.new("RGB", (100, 100)), "unknown")

    def test_tesseract_execution_error_is_not_silenced(self):
        with (
            patch.object(preprocess, "TESSERACT_AVAILABLE", True),
            patch.object(
                preprocess.pytesseract,
                "image_to_data",
                side_effect=RuntimeError("OCR process failed"),
            ),
        ):
            with self.assertRaisesRegex(
                preprocess.OCRProcessingError,
                "Tesseract OCR nije uspio obraditi dokument",
            ):
                preprocess.run_ocr_on_image(Image.new("RGB", (100, 100)), "unknown")

    def test_image_without_recognized_text_has_friendly_error(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / "blank.png"
            Image.new("RGB", (300, 200), "white").save(image_path)
            with (
                patch.object(predict_text_model, "TESSERACT_AVAILABLE", True),
                patch.object(
                    predict_text_model,
                    "run_ocr_on_image",
                    return_value=("", preprocess.empty_ocr_payload("unknown")),
                ),
            ):
                with self.assertRaisesRegex(
                    preprocess.OCRProcessingError,
                    "OCR nije uspio pročitati tekst iz dokumenta",
                ):
                    predict_text_model.extract_text_from_image(image_path)

    def test_pdf_uses_embedded_text_without_ocr(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file:
            document = fitz.open()
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "Ovo je ugrađeni tekst PDF dokumenta koji je dovoljno dug za klasifikaciju.",
            )
            document.save(pdf_file.name)
            document.close()

            with patch.object(
                predict_text_model,
                "run_ocr_on_image",
                side_effect=AssertionError("OCR se ne smije pozvati"),
            ) as run_ocr:
                text = predict_text_model.extract_text_from_pdf(Path(pdf_file.name))

            self.assertGreaterEqual(len(text), preprocess.MIN_TEXT_CHARS)
            run_ocr.assert_not_called()

    def test_scanned_pdf_falls_back_to_ocr(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file:
            document = fitz.open()
            document.new_page()
            document.save(pdf_file.name)
            document.close()

            expected_text = "Tekst pročitan OCR-om iz skeniranog PDF dokumenta."
            with patch.object(
                predict_text_model,
                "run_ocr_on_image",
                return_value=(expected_text, preprocess.empty_ocr_payload("unknown")),
            ) as run_ocr:
                text = predict_text_model.extract_text_from_pdf(Path(pdf_file.name))

            self.assertEqual(text, expected_text)
            run_ocr.assert_called_once()


if __name__ == "__main__":
    unittest.main()
