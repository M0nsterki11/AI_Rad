import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from PIL import Image

from src import document_conversion


class DocumentConversionTests(unittest.TestCase):
    def test_missing_libreoffice_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            docx_path = temporary_path / "document.docx"
            docx_path.write_bytes(b"test")

            with patch.object(
                document_conversion,
                "find_libreoffice_executable",
                return_value=None,
            ):
                with self.assertRaisesRegex(
                    document_conversion.DocumentConversionError,
                    "LibreOffice nije dostupan",
                ):
                    document_conversion.convert_docx_to_pdf(docx_path, temporary_path)

    def test_docx_conversion_uses_headless_libreoffice(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            docx_path = temporary_path / "document.docx"
            docx_path.write_bytes(b"test")
            expected_pdf = temporary_path / "document.pdf"

            def create_pdf(command, **kwargs):
                expected_pdf.write_bytes(b"pdf")
                return subprocess.CompletedProcess(command, 0, "converted", "")

            with (
                patch.object(
                    document_conversion,
                    "find_libreoffice_executable",
                    return_value=Path("libreoffice"),
                ),
                patch.object(
                    document_conversion.subprocess,
                    "run",
                    side_effect=create_pdf,
                ) as run_command,
            ):
                result = document_conversion.convert_docx_to_pdf(
                    docx_path,
                    temporary_path,
                )

            self.assertEqual(result, expected_pdf)
            command = run_command.call_args.args[0]
            self.assertEqual(command[0], "libreoffice")
            self.assertTrue(command[1].startswith("-env:UserInstallation=file:"))
            self.assertIn("--headless", command)
            self.assertIn("--convert-to", command)
            self.assertIn("--outdir", command)
            self.assertEqual(
                run_command.call_args.kwargs["timeout"],
                document_conversion.LIBREOFFICE_TIMEOUT_SECONDS,
            )

    def test_pdf_first_page_is_rendered_to_png(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            pdf_path = temporary_path / "document.pdf"

            document = fitz.open()
            page = document.new_page(width=600, height=800)
            page.insert_text((72, 72), "Visual document for ResNet50")
            document.save(pdf_path)
            document.close()

            image_path = document_conversion.convert_pdf_first_page_to_image(
                pdf_path,
                temporary_path,
            )

            self.assertTrue(image_path.is_file())
            with Image.open(image_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (1200, 1600))


if __name__ == "__main__":
    unittest.main()
