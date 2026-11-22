import unittest
from scripts.check_diagnostic_regressions import checks


class DiagnosticRegressionTests(unittest.TestCase):
    def test_confirmed_regressions(self):
        result = checks()
        for row in result["rotated_pdf_boxes"]:
            self.assertTrue(row["correct"], row)
        self.assertTrue(result["cache"]["unchanged_reuses_object"])
        self.assertTrue(result["cache"]["updated_reloads_object"])
        self.assertTrue(result["xlm_roberta_incomplete_fallback"]["selected_complete_legacy"])
        self.assertTrue(result["layoutlmv3_incomplete_fallback"]["selected_complete_legacy"])
        self.assertTrue(result["resnet_download_aggregation_preserved"])


if __name__ == "__main__":
    unittest.main()
