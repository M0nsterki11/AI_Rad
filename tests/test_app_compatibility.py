import json
import tempfile
import unittest
from pathlib import Path

import app


class AppCompatibilityTests(unittest.TestCase):
    def test_prepared_inputs_make_all_models_ready(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image_path = root / "image.png"
            text_path = root / "text.txt"
            ocr_path = root / "ocr.json"
            image_path.write_bytes(b"image")
            text_path.write_text("dovoljno teksta", encoding="utf-8")
            ocr_path.write_text(
                json.dumps({"words": ["tekst"], "boxes": [[0, 0, 10, 10]]}),
                encoding="utf-8",
            )
            prepared = {
                "image_path": image_path,
                "text_path": text_path,
                "ocr_path": ocr_path,
                "errors": [],
            }

            for model_key in ["resnet50", "xlm_roberta", "layoutlmv3"]:
                ready, reason = app.prepared_model_status(prepared, model_key)
                self.assertTrue(ready, reason)

    def test_docx_without_visual_conversion_only_keeps_xlm_ready(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            text_path = Path(temporary_dir) / "text.txt"
            text_path.write_text("DOCX tekst ostaje dostupan", encoding="utf-8")
            prepared = {
                "image_path": None,
                "text_path": text_path,
                "ocr_path": None,
                "errors": [
                    "Vizualni input nije pripremljen: LibreOffice nije dostupan."
                ],
            }

            self.assertFalse(app.prepared_model_status(prepared, "resnet50")[0])
            self.assertTrue(app.prepared_model_status(prepared, "xlm_roberta")[0])
            self.assertFalse(app.prepared_model_status(prepared, "layoutlmv3")[0])
            self.assertEqual(
                app.run_prepared_model_prediction(prepared, "resnet50")["status"],
                "Preskočeno",
            )
            self.assertEqual(
                app.run_prepared_model_prediction(prepared, "layoutlmv3")["status"],
                "Preskočeno",
            )


if __name__ == "__main__":
    unittest.main()
