import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from docx import Document
from PIL import Image

from src import document_adapter
from src.document_conversion import DocumentConversionError


class UploadedFileStub:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getbuffer(self):
        return memoryview(self._data)


def sample_payload(words=None):
    words = words or ["test", "document"]
    return {
        "label": "unknown",
        "words": words,
        "boxes": [[10, 10 + index * 20, 100, 25 + index * 20] for index, _ in enumerate(words)],
        "confidences": ["95"] * len(words),
        "page_indices": [0] * len(words),
    }


class DocumentAdapterTests(unittest.TestCase):
    def test_txt_prepares_text_image_and_layout(self):
        upload = UploadedFileStub(
            "document.txt",
            "Ovo je dovoljno dugačak tekstualni dokument za sva tri modela.".encode("utf-8"),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            prepared = document_adapter.prepare_document_for_models(upload, temporary_dir)

            self.assertEqual(prepared["suffix"], ".txt")
            self.assertTrue(prepared["text_path"].is_file())
            self.assertTrue(prepared["image_path"].is_file())
            self.assertTrue(prepared["ocr_path"].is_file())
            payload = json.loads(prepared["ocr_path"].read_text(encoding="utf-8"))
            self.assertTrue(payload["words"])
            self.assertEqual(len(payload["words"]), len(payload["boxes"]))

    def test_docx_keeps_text_when_libreoffice_is_unavailable(self):
        docx_buffer = io.BytesIO()
        document = Document()
        document.add_paragraph(
            "DOCX s dovoljno teksta mora ostati dostupan XLM-RoBERTa modelu i bez LibreOfficea."
        )
        document.save(docx_buffer)
        upload = UploadedFileStub("document.docx", docx_buffer.getvalue())

        with (
            tempfile.TemporaryDirectory() as temporary_dir,
            patch.object(
                document_adapter,
                "convert_docx_to_pdf",
                side_effect=DocumentConversionError("LibreOffice nije dostupan"),
            ),
        ):
            prepared = document_adapter.prepare_document_for_models(upload, temporary_dir)

            self.assertTrue(prepared["text_path"].is_file())
            self.assertIsNone(prepared["pdf_path"])
            self.assertIsNone(prepared["image_path"])
            self.assertIsNone(prepared["ocr_path"])
            self.assertTrue(any("Vizualni input" in error for error in prepared["errors"]))

    def test_image_uses_ocr_for_text_and_layout(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (300, 200), "white").save(image_buffer, format="PNG")
        upload = UploadedFileStub("document.png", image_buffer.getvalue())
        ocr_text = "Tekst pročitan OCR-om iz slike za tekstualni i layout model."

        with (
            tempfile.TemporaryDirectory() as temporary_dir,
            patch.object(
                document_adapter,
                "_ocr_image",
                return_value=(ocr_text, sample_payload()),
            ),
        ):
            prepared = document_adapter.prepare_document_for_models(upload, temporary_dir)

            self.assertEqual(prepared["image_path"], prepared["original_path"])
            self.assertIn("Tekst pročitan OCR-om", prepared["text_path"].read_text(encoding="utf-8"))
            self.assertTrue(prepared["ocr_path"].is_file())

    def test_pdf_uses_embedded_text_and_prepares_first_page(self):
        document = fitz.open()
        page = document.new_page(width=600, height=800)
        page.insert_text(
            (72, 72),
            "PDF s ugrađenim tekstom koji je dovoljno dug za XLM-RoBERTa klasifikaciju.",
        )
        pdf_bytes = document.tobytes()
        document.close()
        upload = UploadedFileStub("document.pdf", pdf_bytes)

        def prepare_layout(image_path, fallback_text, ocr_path, errors):
            Path(ocr_path).write_text(json.dumps(sample_payload()), encoding="utf-8")
            return Path(ocr_path), ""

        with (
            tempfile.TemporaryDirectory() as temporary_dir,
            patch.object(
                document_adapter,
                "_prepare_layout_payload",
                side_effect=prepare_layout,
            ),
        ):
            prepared = document_adapter.prepare_document_for_models(upload, temporary_dir)

            self.assertEqual(prepared["pdf_path"], prepared["original_path"])
            self.assertTrue(prepared["image_path"].is_file())
            self.assertTrue(prepared["text_path"].is_file())
            self.assertTrue(prepared["ocr_path"].is_file())


if __name__ == "__main__":
    unittest.main()
