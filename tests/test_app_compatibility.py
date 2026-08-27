import unittest

import app


class AppCompatibilityTests(unittest.TestCase):
    def test_resnet_accepts_docx_for_visual_conversion(self):
        compatible, incompatible = app.split_compatible_models(["resnet50"], ".docx")
        self.assertEqual(compatible, ["resnet50"])
        self.assertEqual(incompatible, [])

    def test_all_models_use_resnet_and_xlm_for_docx(self):
        compatible, incompatible = app.split_compatible_models(
            ["resnet50", "xlm_roberta", "layoutlmv3"],
            ".docx",
        )
        self.assertEqual(compatible, ["resnet50", "xlm_roberta"])
        self.assertEqual(incompatible, ["layoutlmv3"])

    def test_resnet_still_rejects_txt(self):
        compatible, incompatible = app.split_compatible_models(["resnet50"], ".txt")
        self.assertEqual(compatible, [])
        self.assertEqual(incompatible, ["resnet50"])
        self.assertEqual(
            app.incompatible_model_reason("resnet50", ".txt"),
            "ResNet50 je vizualni model i ne podržava TXT bez pretvaranja u sliku.",
        )


if __name__ == "__main__":
    unittest.main()
