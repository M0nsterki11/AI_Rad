import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_expansion import (
    CLASS_NAMES,
    DatasetExpansionError,
    FingerprintIndex,
    build_fingerprint_record,
    group_aware_stratified_split,
    validate_training_path,
)
from scripts.expand_existing_dataset_realworld import (
    AcceptedDocument,
    execute_worker_command,
    iter_huggingface_rows_api,
    load_resume_documents,
    load_staging_manifest,
    processed_outputs_valid,
    stage_augmentations,
    staging_satisfies_plan,
    streaming_shuffle_buffer_size,
    write_staging_manifest,
)


class DatasetExpansionTests(unittest.TestCase):
    def test_exact_file_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("A sufficiently long contract paragraph for hashing.", encoding="utf-8")
            second.write_bytes(first.read_bytes())

            index = FingerprintIndex()
            original = build_fingerprint_record(
                key="first", raw_path=first, label="contract", source="fixture"
            )
            candidate = build_fingerprint_record(
                key="second", raw_path=second, label="contract", source="fixture"
            )
            index.add(original)

            match = index.find_duplicate(candidate)
            self.assertIsNotNone(match)
            self.assertEqual(match.reason, "identical_sha256")

    def test_normalized_text_duplicate_is_rejected_across_different_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text(
                "This Agreement applies to every party and remains effective for one year.",
                encoding="utf-8",
            )
            second.write_text(
                "THIS agreement applies to every party, and remains effective for one year!",
                encoding="utf-8",
            )

            index = FingerprintIndex()
            original = build_fingerprint_record(
                key="first", raw_path=first, label="contract", source="fixture"
            )
            candidate = build_fingerprint_record(
                key="second", raw_path=second, label="contract", source="fixture"
            )
            index.add(original)

            match = index.find_duplicate(candidate)
            self.assertIsNotNone(match)
            self.assertEqual(match.reason, "identical_normalized_text")

    def test_visual_fingerprint_is_created_for_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document.png"
            Image.new("RGB", (640, 900), "white").save(path)
            record = build_fingerprint_record(
                key="image", raw_path=path, label="invoice", source="fixture"
            )
            self.assertIsNotNone(record.image_phash)
            self.assertEqual(record.image_width, 640)
            self.assertEqual(record.image_height, 900)

    def test_augmentation_stays_with_parent_in_group_aware_split(self):
        rows = []
        for label in CLASS_NAMES:
            for index in range(20):
                document_id = f"{label}_{index:04d}"
                rows.append(
                    {
                        "id": document_id,
                        "label": label,
                        "raw_path": f"data/raw/{label}/{document_id}.pdf",
                        "image_path": f"data/processed/images/{document_id}.png",
                        "text_path": f"data/processed/texts/{document_id}.txt",
                        "ocr_path": f"data/processed/ocr/{document_id}.json",
                    }
                )
            child_id = f"{label}_extra_0001"
            rows.append(
                {
                    "id": child_id,
                    "label": label,
                    "raw_path": f"data/raw/{label}/{child_id}.jpg",
                    "image_path": f"data/processed/images/{child_id}.png",
                    "text_path": f"data/processed/texts/{child_id}.txt",
                    "ocr_path": f"data/processed/ocr/{child_id}.json",
                }
            )

        parents = {f"{label}_extra_0001": f"{label}_0000" for label in CLASS_NAMES}
        splits = group_aware_stratified_split(rows, parents, seed=42)
        split_by_id = {
            row["id"]: split_name
            for split_name, split_rows in splits.items()
            for row in split_rows
        }
        for child, parent in parents.items():
            self.assertEqual(split_by_id[child], split_by_id[parent])
        self.assertEqual(sum(len(items) for items in splits.values()), len(rows))

    def test_external_test_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forbidden = root / "data" / "external_test" / "invoice" / "sample.pdf"
            forbidden.parent.mkdir(parents=True)
            forbidden.touch()
            with self.assertRaises(DatasetExpansionError):
                validate_training_path(root, forbidden)

    def test_processed_output_validation_checks_all_three_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            text_path = root / "sample.txt"
            ocr_path = root / "sample.json"
            Image.new("RGB", (100, 120), "white").save(image_path)
            text_path.write_text("Readable document text for resume validation.", encoding="utf-8")
            ocr_path.write_text(
                json.dumps({"words": ["Readable"], "boxes": [[1, 2, 20, 10]]}),
                encoding="utf-8",
            )

            valid, error = processed_outputs_valid((image_path, text_path, ocr_path))
            self.assertTrue(valid, error)
            ocr_path.write_text(json.dumps({"words": ["x"], "boxes": []}), encoding="utf-8")
            valid, _ = processed_outputs_valid((image_path, text_path, ocr_path))
            self.assertFalse(valid)

    def test_staging_manifest_round_trip_preserves_resume_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            raw_path = staging / "raw" / "email" / "email_extra_0001.png"
            raw_path.parent.mkdir(parents=True)
            Image.new("RGB", (100, 120), "white").save(raw_path)
            final_path = PROJECT_ROOT / "data" / "raw" / "email" / raw_path.name
            document = AcceptedDocument(
                document_id="email_extra_0001",
                label="email",
                source_name="fixture",
                source_locator="fixture://dataset",
                original_id="source-1",
                raw_staging_path=raw_path,
                raw_final_path=final_path,
                text_hint="Subject: Test\n\nThis is a sufficiently long message body.",
            )

            write_staging_manifest([document], staging)
            loaded = load_staging_manifest(staging)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].document_id, document.document_id)
            self.assertEqual(loaded[0].text_hint, document.text_hint)

    def test_resume_preserves_manifest_rows_and_adds_only_untracked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            first_path = staging / "raw" / "email" / "email_extra_0001.png"
            second_path = staging / "raw" / "email" / "email_extra_0002.png"
            first_path.parent.mkdir(parents=True)
            Image.new("RGB", (100, 120), "white").save(first_path)
            Image.new("RGB", (100, 120), "white").save(second_path)
            first = AcceptedDocument(
                document_id="email_extra_0001",
                label="email",
                source_name="source_a",
                source_locator="fixture://a",
                original_id="preserve-this-origin",
                raw_staging_path=first_path,
                raw_final_path=PROJECT_ROOT / "data" / "raw" / "email" / first_path.name,
                text_hint="",
            )
            write_staging_manifest([first], staging)
            plan = {
                label: {
                    "real_target": 0,
                    "quotas": {},
                    "augmentation_target": 0,
                }
                for label in CLASS_NAMES
            }
            plan["email"] = {
                "real_target": 2,
                "quotas": {"source_a": 1, "source_b": 1},
                "augmentation_target": 0,
            }
            sources = [
                {"name": "source_a", "url": "fixture://a"},
                {"name": "source_b", "url": "fixture://b"},
            ]

            loaded = load_resume_documents(staging, plan, sources)
            by_id = {document.document_id: document for document in loaded}

            self.assertEqual(len(loaded), 2)
            self.assertEqual(by_id["email_extra_0001"].original_id, "preserve-this-origin")
            self.assertEqual(by_id["email_extra_0002"].source_name, "source_b")

    def test_worker_command_times_out_and_is_terminated(self):
        timed_out, _, _, _, elapsed = execute_worker_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=0.2,
        )
        self.assertTrue(timed_out)
        self.assertLess(elapsed, 8)

    def test_streaming_shuffle_buffer_is_memory_bounded(self):
        self.assertEqual(streaming_shuffle_buffer_size(1), 1)
        self.assertEqual(streaming_shuffle_buffer_size(3), 3)
        self.assertEqual(streaming_shuffle_buffer_size(1728), 4)

    def test_partial_staging_does_not_satisfy_plan(self):
        plan = {
            label: {"quotas": {}, "augmentation_target": 0}
            for label in CLASS_NAMES
        }
        plan["email"] = {
            "quotas": {"email_source": 2},
            "augmentation_target": 1,
        }

        def document(document_id: str, *, augmented: bool = False) -> AcceptedDocument:
            path = Path("staging") / f"{document_id}.png"
            return AcceptedDocument(
                document_id=document_id,
                label="email",
                source_name="email_source",
                source_locator="fixture://email",
                original_id=document_id,
                raw_staging_path=path,
                raw_final_path=path,
                text_hint="",
                is_augmented=augmented,
                parent_id="email_extra_0001" if augmented else "",
            )

        real_documents = [document("email_extra_0001"), document("email_extra_0002")]
        self.assertFalse(staging_satisfies_plan(real_documents[:1], plan))
        self.assertFalse(staging_satisfies_plan(real_documents, plan))
        self.assertTrue(
            staging_satisfies_plan(
                real_documents + [document("email_extra_0003", augmented=True)],
                plan,
            )
        )

    def test_huggingface_rows_api_materializes_a_bounded_page(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (128, 128), "white").save(image_buffer, format="PNG")
        encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class TransientFailure:
            status_code = 502

            def raise_for_status(self):
                raise requests.HTTPError("temporary gateway error", response=self)

        responses = [
            TransientFailure(),
            Response({"num_rows_total": 1}),
            Response(
                {
                    "rows": [
                        {
                            "row_idx": 7,
                            "row": {
                                "class_name": "email",
                                "image_base64": encoded,
                            },
                        }
                    ]
                }
            ),
        ]
        source = {
            "name": "fixture_rows_api",
            "label": "email",
            "dataset": "fixture/dataset",
            "split": "train",
            "accepted_labels": ["email"],
            "materialize": "original",
            "url": "https://example.test/dataset",
        }
        with (
            patch(
                "scripts.expand_existing_dataset_realworld.requests.get",
                side_effect=responses,
            ) as mocked_get,
            patch("scripts.expand_existing_dataset_realworld.time.sleep"),
        ):
            candidates = list(
                iter_huggingface_rows_api(source, scan_limit=1, seed=42, timeout=5)
            )

        self.assertEqual(mocked_get.call_count, 3)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].original_id, "row_00000007")
        self.assertEqual(candidates[0].extension, ".png")

    def test_augmentation_uses_a_replacement_after_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            originals = []
            for index in range(5):
                path = staging / "raw" / "email" / f"email_extra_{index + 1:04d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (160, 200), (240 - index, 240, 240)).save(path)
                originals.append(
                    AcceptedDocument(
                        document_id=path.stem,
                        label="email",
                        source_name="fixture",
                        source_locator="fixture://email",
                        original_id=f"origin-{index}",
                        raw_staging_path=path,
                        raw_final_path=PROJECT_ROOT / "data" / "raw" / "email" / path.name,
                        text_hint="",
                    )
                )

            plan = {
                label: {"augmentation_target": 0}
                for label in CLASS_NAMES
            }
            plan["email"]["augmentation_target"] = 1

            class Fingerprints:
                def __init__(self):
                    self.calls = 0

                def find_duplicate(self, *_args, **_kwargs):
                    self.calls += 1
                    if self.calls == 1:
                        return SimpleNamespace(
                            reason="near_identical_first_page",
                            similar_to="fixture-parent",
                            similarity_score=1.0,
                        )
                    return None

                def add(self, _fingerprint):
                    return None

            augmented, skipped = stage_augmentations(
                plan=plan,
                accepted_real=originals,
                metadata_rows=[],
                staging_root=staging,
                fingerprint_index=Fingerprints(),
                duplicate_rows=[],
                args=SimpleNamespace(seed=42, augmentation_fraction=0.20),
            )

            self.assertEqual(len(augmented), 1)
            self.assertEqual(skipped["email"], 1)


if __name__ == "__main__":
    unittest.main()
