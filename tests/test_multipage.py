import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
import torch
from PIL import Image

from src.multipage import (
    ManifestLeakageError,
    aggregate_document_predictions,
    aggregate_scores,
    choose_aggregation_method,
    document_balanced_weights,
    normalize_box_to_1000,
    select_representative_indices,
    tokenize_document_chunks,
    validate_document_manifest,
)
from src.document_adapter import prepare_document_for_models
from src.multipage_manifest import build_document_manifest
from src.multipage_preprocess import (
    LAYOUT_STATUS_EMPTY,
    LAYOUT_STATUS_VALID,
    PageArtifact,
    prepare_pdf_page_artifacts,
)


class FakeTokenizer:
    def __call__(
        self,
        text,
        *,
        truncation,
        max_length,
        stride,
        return_overflowing_tokens,
        return_attention_mask,
        padding,
    ):
        del text, truncation, stride, return_overflowing_tokens, return_attention_mask, padding
        chunks = [list(range(max_length)) for _ in range(20)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * max_length for _ in chunks],
        }


class MultipageTests(unittest.TestCase):
    def test_short_document_selects_every_page(self):
        self.assertEqual(select_representative_indices(12), list(range(12)))

    def test_long_document_selects_head_middle_and_tail(self):
        selected = select_representative_indices(47)
        self.assertEqual(len(selected), 12)
        self.assertEqual(selected[:4], [0, 1, 2, 3])
        self.assertEqual(selected[-3:], [44, 45, 46])
        self.assertEqual(selected, sorted(set(selected)))
        self.assertEqual(len(selected[4:-3]), 5)

    def test_tokenizer_overflow_is_limited_with_representative_chunks(self):
        chunks = tokenize_document_chunks(FakeTokenizer(), "document", max_chunks=12)
        self.assertEqual(len(chunks), 12)
        self.assertEqual([row["chunk_index"] for row in chunks[:4]], [0, 1, 2, 3])
        self.assertEqual([row["chunk_index"] for row in chunks[-3:]], [17, 18, 19])
        self.assertTrue(all(len(row["input_ids"]) == 512 for row in chunks))

    def test_aggregation_modes_return_probabilities(self):
        logits = torch.tensor([[4.0, 1.0], [2.0, 3.0], [3.0, 0.0]])
        for method in ("mean", "max", "top_k_mean"):
            _, probabilities = aggregate_scores(logits, method=method, top_k=2)
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
            self.assertEqual(tuple(probabilities.shape), (2,))

    def test_aggregation_tie_prefers_configured_default(self):
        method, _ = choose_aggregation_method(
            {"doc": [[1.0, 0.0]]},
            {"doc": 0},
            class_count=2,
        )
        self.assertEqual(method, "top_k_mean")

    def test_pixel_boxes_are_normalized_with_shared_logic(self):
        self.assertEqual(normalize_box_to_1000([10, 20, 100, 200], 200, 400), [50, 50, 500, 500])

    def test_document_level_evaluation_counts_each_document_once(self):
        result = aggregate_document_predictions(
            {
                "doc-a": [[4.0, 1.0], [3.0, 1.0]],
                "doc-b": [[1.0, 4.0], [1.0, 3.0], [0.0, 5.0]],
            },
            {"doc-a": 0, "doc-b": 1},
            class_count=2,
            method="top_k_mean",
        )
        self.assertEqual(result["documents_evaluated"], 2)
        self.assertEqual(result["accuracy"], 1.0)

    def test_document_balanced_weights_equalize_document_mass(self):
        rows = [
            {"document_id": "short"},
            {"document_id": "long"},
            {"document_id": "long"},
            {"document_id": "long"},
        ]
        weights = document_balanced_weights(rows)
        self.assertEqual(weights, [1.0, 1 / 3, 1 / 3, 1 / 3])

    def test_manifest_rejects_group_leakage(self):
        rows = [
            {
                "document_id": "original",
                "parent_document_id": "original",
                "augmentation_group_id": "group-1",
                "label": "invoice",
                "raw_path": "data/raw/invoice/original.pdf",
                "split": "train",
            },
            {
                "document_id": "augmented",
                "parent_document_id": "original",
                "augmentation_group_id": "group-1",
                "label": "invoice",
                "raw_path": "data/raw/invoice/augmented.pdf",
                "split": "test",
            },
        ]
        with self.assertRaises(ManifestLeakageError):
            validate_document_manifest(rows)

    def test_pdf_page_artifacts_are_aligned_one_page_at_a_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "long.pdf"
            document = fitz.open()
            for page_index in range(13):
                page = document.new_page(width=595, height=842)
                page.insert_text((72, 72), f"UNIQUE_PAGE_{page_index}")
            document.save(pdf_path)
            document.close()

            row = {
                "document_id": "doc-1",
                "parent_document_id": "doc-1",
                "augmentation_group_id": "doc-1",
                "label": "scientific",
                "split": "train",
            }
            total, selected, artifacts = prepare_pdf_page_artifacts(
                pdf_path, root / "output", row
            )

            self.assertEqual(total, 13)
            self.assertEqual(len(selected), 12)
            self.assertEqual(len(artifacts), 12)
            for artifact in artifacts:
                payload = json.loads(artifact.ocr_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["document_id"], "doc-1")
                self.assertEqual(payload["page_index"], artifact.page_index)
                self.assertEqual(len(payload["words"]), len(payload["boxes"]))
                self.assertIn(f"UNIQUE_PAGE_{artifact.page_index}", payload["words"])

    def test_blank_pdf_page_is_preserved_for_resnet_and_skipped_for_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "partially_blank.pdf"
            document = fitz.open()
            first_page = document.new_page(width=595, height=842)
            first_page.insert_text((72, 72), "VALID_PAGE_TEXT")
            document.new_page(width=595, height=842)
            document.save(pdf_path)
            document.close()

            row = {
                "document_id": "doc-partial",
                "parent_document_id": "doc-partial",
                "augmentation_group_id": "doc-partial",
                "label": "scientific",
                "split": "train",
            }
            empty_ocr = ("", {"words": [], "boxes": []})
            with patch("src.multipage_preprocess.run_ocr_on_image", return_value=empty_ocr):
                total, selected, artifacts = prepare_pdf_page_artifacts(
                    pdf_path, root / "output", row
                )

            self.assertEqual(total, 2)
            self.assertEqual(selected, [0, 1])
            self.assertEqual(len(artifacts), 2)
            self.assertEqual(artifacts[0].layout_status, LAYOUT_STATUS_VALID)
            self.assertEqual(artifacts[1].layout_status, LAYOUT_STATUS_EMPTY)
            self.assertTrue(artifacts[1].image_path.is_file())
            self.assertTrue(artifacts[1].ocr_path.is_file())
            payload = json.loads(artifacts[1].ocr_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["words"], [])
            self.assertEqual(payload["boxes"], [])
            self.assertEqual(payload["layout_status"], LAYOUT_STATUS_EMPTY)

    def test_page_builder_continues_after_document_failure(self):
        from scripts import build_multipage_dataset as builder

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            ocr_path = root / "page.json"
            Image.new("RGB", (100, 100), "white").save(image_path)
            ocr_path.write_text(
                json.dumps({"words": ["valid"], "boxes": [[1, 1, 20, 20]]}),
                encoding="utf-8",
            )
            rows = []
            for document_id in ("broken", "valid"):
                rows.append(
                    {
                        "document_id": document_id,
                        "parent_document_id": document_id,
                        "augmentation_group_id": document_id,
                        "label": "contract",
                        "split": "train",
                        "raw_path": f"data/raw/contract/{document_id}.pdf",
                        "text_path": f"data/processed/texts/{document_id}.txt",
                    }
                )
            artifact = PageArtifact(
                document_id="valid",
                parent_document_id="valid",
                augmentation_group_id="valid",
                label="contract",
                split="train",
                page_index=0,
                total_pages=1,
                image_path=image_path,
                ocr_path=ocr_path,
                words=["valid"],
                boxes=[[1, 1, 20, 20]],
                extraction_method="pdf_embedded_words",
            )
            side_effect = [RuntimeError("bad PDF"), (1, [0], [artifact])]
            with patch.object(builder, "PROJECT_ROOT", root), patch.object(
                builder, "prepare_document_page_artifacts", side_effect=side_effect
            ):
                page_rows, failed_pages, failed_documents, stats = (
                    builder.build_page_artifacts(rows, root / "multipage", False)
                )

            self.assertEqual(len(page_rows), 1)
            self.assertEqual(page_rows[0]["document_id"], "valid")
            self.assertEqual(failed_pages, [])
            self.assertEqual(len(failed_documents), 1)
            self.assertEqual(failed_documents[0]["document_id"], "broken")
            self.assertEqual(stats["documents_with_valid_layout_page"], 1)
            self.assertEqual(stats["documents_without_valid_layout_page"], 1)

    def test_live_adapter_prepares_one_shared_multipage_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "uploaded.pdf"
            document = fitz.open()
            for page_index in range(13):
                page = document.new_page(width=595, height=842)
                page.insert_text((72, 72), f"ADAPTER_PAGE_{page_index}")
            document.save(pdf_path)
            document.close()

            prepared = prepare_document_for_models(pdf_path, root / "prepared")
            self.assertEqual(prepared["total_pages"], 13)
            self.assertEqual(len(prepared["analyzed_page_indices"]), 12)
            self.assertEqual(len(prepared["page_artifacts"]), 12)
            self.assertIn("ADAPTER_PAGE_12", prepared["text_path"].read_text(encoding="utf-8"))
            for page in prepared["page_artifacts"]:
                self.assertEqual(len(page["words"]), len(page["boxes"]))

    def test_training_entry_points_do_not_reference_legacy_splits(self):
        project_root = Path(__file__).resolve().parents[1]
        forbidden = ("SPLITS_DIR", "train_test_split", "stratified_split", "data/splits")
        for filename in ("train_resnet.py", "train_text_model.py", "train_layoutlm.py"):
            source = (project_root / "src" / filename).read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, source, f"{filename} contains {marker}")

    def test_manifest_builder_keeps_augmentation_with_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = []
            for label in ("invoice", "cv", "contract", "email", "scientific"):
                class_dir = root / "data" / "raw" / label
                class_dir.mkdir(parents=True)
                for index in range(20):
                    document_id = f"{label}_{index:04d}"
                    raw_path = class_dir / f"{document_id}.txt"
                    raw_path.write_text(f"{label} unique document {index}", encoding="utf-8")
                    metadata.append(
                        {
                            "id": document_id,
                            "label": label,
                            "raw_path": str(raw_path.relative_to(root)),
                            "image_path": f"data/processed/images/{document_id}.png",
                            "text_path": f"data/processed/texts/{document_id}.txt",
                            "ocr_path": f"data/processed/ocr/{document_id}.json",
                        }
                    )

            child_id = "invoice_aug_0001"
            child_path = root / "data" / "raw" / "invoice" / f"{child_id}.txt"
            child_path.write_text("augmented invoice content", encoding="utf-8")
            metadata.append(
                {
                    "id": child_id,
                    "label": "invoice",
                    "raw_path": str(child_path.relative_to(root)),
                    "image_path": f"data/processed/images/{child_id}.png",
                    "text_path": f"data/processed/texts/{child_id}.txt",
                    "ocr_path": f"data/processed/ocr/{child_id}.json",
                }
            )
            sources = [
                {
                    "id": child_id,
                    "original_id": "invoice_0000",
                    "is_augmented": "True",
                }
            ]

            manifest = build_document_manifest(root, metadata, sources, seed=42)
            by_id = {row["document_id"]: row for row in manifest}
            self.assertEqual(
                by_id[child_id]["split"], by_id["invoice_0000"]["split"]
            )
            self.assertEqual(
                by_id[child_id]["augmentation_group_id"],
                by_id["invoice_0000"]["augmentation_group_id"],
            )


if __name__ == "__main__":
    unittest.main()
